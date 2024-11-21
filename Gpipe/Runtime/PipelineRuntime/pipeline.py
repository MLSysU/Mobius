import torch
import torch.distributed as dist
from transformer.model.transformer_lm import TransformerLM
import copy
from torchviz import make_dot
from Runtime import *
from utils import *
import torch.nn.functional as F
import time

def print_grad(grad):
    print("Grad:", grad)
    return 


class Pipeline():
    def __init__(self,args,module_list:list,world_size:int,global_rank:int,local_rank:int,embedding_layer,train_batches,norm_layer,lm_head):
        self.module_list=module_list
        self.src_list=[] # inputs fetched by stage0
        self.activation_list=[[] for _ in range(args.num_stages)] # 一个gpu可能负责多个stage,每个stage又有很多micro_batch的activation,所以写成长度为num_stages的数组
        self.grad_list=[[] for _ in range(args.num_stages)] 
        self.input_list=[[] for _ in range(args.num_stages)] # 虽然和activation_list中的元素是一样的东西，但是因为在my_stage也无法确定input是来自哪个stage的activation,所以单独写一个input_list
        self.batch_size=args.batch_size
        self.num_chunks=args.num_chunks
        self.seq_length=args.seq_length
        self.embedding_dim=args.embedding_dim
        self.ff_dim=args.ff_dim
        self.world_size=world_size
        self.num_stages=args.num_stages
        self.my_rank=global_rank
        self.local_rank=local_rank
        self.input_chunk_id=0
        self.last_send=None # 上一次异步发送是否成功结束的标志
        self.last_receive=None # 上一次异步接收数据是否成功的标志
        self.dtype=torch.float32
        self.total_parameters=0
        self.optimizer=self.construct_optimizer()
        self.input_data = [[] for _ in range(args.num_iterations)]
        self.iteration=-1
        self.device=torch.device(f'cuda:{local_rank}')
        self.last_recv_tensor=torch.zeros([self.batch_size//self.num_chunks,self.seq_length,self.embedding_dim],dtype=self.dtype,device=self.device)
        self.train_batches=train_batches
        self.embedding_layer=embedding_layer
        self.norm_layer=norm_layer.to(self.device)
        self.lm_head=lm_head.to(self.device)
        self.state_dict = [{} for _ in range(self.num_stages)] 
        self.module=None
        self.local_module_list=[]
        self.load_stream=torch.cuda.Stream()
        self.comm_stream=torch.cuda.Stream()
        self.comp_stream=torch.cuda.Stream()
        self.offload_stream=torch.cuda.Stream()

    def construct_optimizer(self):
        parameters=[]
        for module in self.module_list:
            module_parameter=list(module.parameters())  # parameters()是nn.Module自带的函数，返回一个生成器，可以迭代调用出模型每一层的参数大小
            parameters+=module_parameter
            self.total_parameters+=sum(p.numel() for p in module_parameter)
        return torch.optim.Adam(parameters,lr=0.0001,weight_decay=1e-3)
        



    def parse_action(self,action:str) ->tuple:
        """
        The format should be '{action} {source_stage_id} {my_stage_id} {target_stage_id} {chunk_id}'.
        while the action should be within [generate_data,get_data,forward_first,forward_middle,
        forward_last,backward_first,backward_middle,backward_last].
        my_stage_id refers to the id of the current stage.
        target_stage_id refers to the id of the other stage which my_stage sends data to.
        source_stage_id refers to the id of the other stage which sends data to my_stage.
        The parameter should be set to -1 if not related to this action.
        """
        assert action.count(" ") ==4, "There should be five parameters" # 逗号后面跟着的内容意思是不符合assert条件时输出的东西
        action=action.split()
        action_name=action[0]
        source_stage_id=int(action[1])
        my_stage_id=int(action[2])
        target_stage_id=int(action[3])
        chunk_id=int(action[4])
        return action_name,source_stage_id,my_stage_id,target_stage_id,chunk_id


        
    '''
    下面写的函数都是每个stage完整的行为
    '''

    # 1.第一个stage需要加载数据，前向计算，向下一个stage发送计算结果[generate_data,forward_first1,f12,f13,f14]
    # 2.中间的stage需要从上个stage那里拿到计算结果，前向计算，向下一个stage发送计算结果
    # 3.最后一个stage需要从上个stage那里拿到计算结果，前向计算

    def forward_first(self,my_stage_id:int,target_stage_id:int,chunk_id:int):     
        # get data
        input_data=self.get_data()
        # embedding layer
        self.embedding_layer.to(self.device)
        input_data=self.embedding_layer(input_data)
        self.input_data[self.iteration].append(input_data)
        self.input_list[my_stage_id].append(input_data)
        # with torch.cuda.stream(self.load)
        # still working...

        # compute
        result_tensor=self.forward_compute(input_data,my_stage_id,chunk_id)
        self.activation_list[my_stage_id].append(result_tensor)
        # send activation
        target_rank=target_stage_id%self.world_size
        self.send_activation(target_rank,result_tensor)
        # make_dot(result_tensor, params={"input_data": input_data}).render("graph", format="png")
        return 

    def forward_middle(self,source_stage_id:int,my_stage_id:int,target_stage_id:int,chunk_id:int):
        # receive data
        source_rank=source_stage_id%self.world_size
        self.receive_activation(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_data=self.last_recv_tensor.clone()
        input_data.requires_grad_(True)
        self.input_list[my_stage_id].append(input_data)
        # compute
        result_tensor=self.forward_compute(input_data,my_stage_id,chunk_id)
        self.activation_list[my_stage_id].append(result_tensor)
        # send activation
        target_rank=target_stage_id%self.world_size
        self.send_activation(target_rank,result_tensor)
        return
    
    def forward_last(self,source_stage_id:int,my_stage_id:int,chunk_id:int):
        # receive data
        source_rank=source_stage_id%self.world_size
        self.receive_activation(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_data=self.last_recv_tensor.clone()
        input_data.requires_grad_(True)
        self.input_list[my_stage_id].append(input_data)
        # forward compute
        result_tensor=self.forward_compute(input_data,my_stage_id,chunk_id)
        self.activation_list[my_stage_id].append(result_tensor)
        # print("pipeline output ",my_stage_id,self.iteration,self.input_chunk_id,result_tensor)
        return 


    # 第一个stage需要从下一个stage接收grad,并且利用grad、本stage的activation和W计算自己的grad
    # 中间的stage需要从下一个stage接收grad,并且计算自己的的grad，将自己的输入的grad发送给前一个stage
    # 最后一个stage利用自己的activation计算LOSS，再计算自己的grad,将自己的输入的grad发送给前一个stage

    def backward_first(self,my_stage_id:int,source_stage_id:int,chunk_id:int):
        # receive data
        source_rank=source_stage_id%self.world_size
        self.receive_grad(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_grad=self.last_recv_tensor.clone()
        # backward compute
        input_tensor=self.input_list[my_stage_id].pop()
        input_tensor.retain_grad()
        # 对于多个microbatch,后前向的先后向，符合栈的顺序
        my_activation=self.activation_list[my_stage_id].pop()
        self.backward_compute(my_activation,my_stage_id,chunk_id,input_grad) 
        # print("send input grad ",my_stage_id,input_tensor.grad.shape,input_tensor.grad)
        return

    def backward_middle(self,source_stage_id:int,my_stage_id:int,target_stage_id:int,chunk_id:int):
        # receive data
        source_rank=source_stage_id%self.world_size
        self.receive_grad(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_grad=self.last_recv_tensor.clone()
        # backward compute
        input_tensor=self.input_list[my_stage_id].pop()
        input_tensor.retain_grad()
        my_activation=self.activation_list[my_stage_id].pop()
        self.backward_compute(my_activation,my_stage_id,chunk_id,input_grad)
        # send input.grad
        target_rank=target_stage_id%self.world_size
        self.send_grad(input_tensor.grad,target_rank)
        # if my_stage_id==1:
        #     print("send input grad ",my_stage_id,input_tensor.grad.shape,input_tensor.grad)
        return 

    def backward_last(self,my_stage_id:int,target_stage_id:int,chunk_id:int):
        if chunk_id==0:
            self.iteration+=1
        # compute the gradient of input
        input_tensor=self.input_list[my_stage_id].pop()
        input_tensor.retain_grad()
        my_activation=self.activation_list[my_stage_id].pop()
        self.backward_compute(my_activation,my_stage_id,chunk_id)
        # send input.grad
        target_rank=target_stage_id%self.world_size
        self.send_grad(input_tensor.grad,target_rank)
        # print("input grad ",my_stage_id,input_tensor.grad)
        return

  

    '''
    下面写的函数都是stage完整行为中可能用到的片段操作
    '''
    def forward_compute(self,input_tensor:torch.tensor,my_stage_id:int,chunk_id:int):    
        # load module
        if chunk_id==0:
            with torch.cuda.stream(self.load_stream):
                # load my model
                # if myself has been prefetched
                if len(self.local_module_list)>my_stage_id//self.world_size and self.local_module_list[my_stage_id//self.world_size][1]=='gpu':
                    self.module=self.local_module_list[my_stage_id//self.world_size][0]
                else:
                    # The first iteration, we need to load the model from the global model list
                    if len(self.local_module_list)<self.num_stages//self.world_size:
                        self.module=copy.deepcopy(self.module_list[my_stage_id])
                        self.module.to(self.device)
                        self.local_module_list.append([self.module,'gpu'])
                    # The following iterations, we need to load the model from the local model list
                    else:
                        self.module=self.local_module_list[my_stage_id//self.world_size][0]
                        load(self.module,self.module_list[my_stage_id])
                        self.local_module_list[my_stage_id//self.world_size][1]='gpu'
                # prefetch model
                if my_stage_id+self.world_size<self.num_stages:
                    next_stage_id=my_stage_id+self.world_size
                    if len(self.local_module_list)<self.num_stages//self.world_size:
                        next_module=copy.deepcopy(self.module_list[next_stage_id])
                        next_module.to(self.device)
                        self.local_module_list.append([next_module,'gpu'])
                    else:
                        next_module=self.local_module_list[next_stage_id//self.world_size][0]
                        load(next_module,self.module_list[next_stage_id])
                        self.local_module_list[next_stage_id//self.world_size][1]='gpu'
        # module.register_forward_pre_hook(reload)

        # compute
        activation=self.module(input_tensor)

        # offload
        if chunk_id==self.num_chunks-1:
            offload(self.module,self.module_list[my_stage_id])
            self.local_module_list[my_stage_id//self.world_size][1]='cpu'
        return activation


    def backward_compute(self,activation:torch.tensor,my_stage_id:int,chunk_id:int,accu_grad=None):
        # 按照数学公式:计算权重的梯度需要下一个stage发送来的累积梯度+本stage的activation
        # Loss/偏w=下一个stage发送来的累积*(偏activation/偏w)
        # 下一个阶段发送来的梯度累计是关于下一层的输入的梯度
        # 当backward()只有一个参数时，那个参数必须是标量；如果有grad_tensor时，activation可以是tensor类型
        if chunk_id==0:
            with torch.cuda.stream(self.load_stream):
                # load_model
                if self.local_module_list[my_stage_id//self.world_size][1]=='cpu':
                    self.module=self.local_module_list[my_stage_id//self.world_size][0]
                    load(self.module,self.module_list[my_stage_id])
                    self.local_module_list[my_stage_id//self.world_size][1]='gpu'

                # prefetch model
                if my_stage_id-self.world_size>=0:
                    last_stage_id=my_stage_id-self.world_size
                    last_module=self.local_module_list[last_stage_id//self.world_size][0]
                    load(last_module,self.module_list[last_stage_id])
                    self.local_module_list[last_stage_id//self.world_size][1]='gpu'

        if accu_grad is None:
            output=self.norm_layer(activation)
            logits=self.lm_head(output[:, -32:, :])
            logits = logits.view(-1, logits.size(-1))
            correct_result=self.train_batches[self.iteration*self.num_chunks+chunk_id]["labels"].to(self.device)
            correct_result=correct_result.view(-1)
            torch.autograd.backward(F.cross_entropy(logits, correct_result))
            # torch.autograd.backward(activation.mean)
        else:
            torch.autograd.backward(activation,grad_tensors=accu_grad)
        # offload model
        if chunk_id==self.num_chunks-1:
            offload(self.module,self.module_list[my_stage_id])
            self.local_module_list[my_stage_id//self.world_size][1]='cpu'

        return 


    def send_activation(self,target_rank:int,activation:torch.tensor):
        # 异步发送
        # 同步发送指的是发送操作执行结束后才能执行后续代码 
        if self.last_send is not None:
            self.last_send.wait()
        send=dist.isend(tensor=activation,dst=target_rank)
        # print("send activation from {} to {}".format(self.my_rank,target_rank))
        self.last_send=send
        return


    def receive_activation(self,source_rank:int):
        # 异步接收
        if self.last_receive is not None:
            self.last_receive.wait()
        # 这是一个非阻塞接收函数，表示进程从另一个源设备接收一个张量的数据。
        # receive 是一个 torch.distributed.Work 对象.
        # receive 可以用于检查该异步通信是否完成。你可以使用 req1.wait() 来等待该通信完成，或者使用 req1.is_completed() 检查通信是否已经结束，而不必等待。        
        receive=dist.irecv(tensor=self.last_recv_tensor,src=source_rank) # 这里是将self.last_recv_tensor指向的对象直接改变，所以如果用self.last_recv_tensor给一个东西赋值需要注意用.clone()
        # print("recv activation from {} to me {}".format(source_rank,self.my_rank))
        self.last_receive=receive
        
        return 
        


    def send_grad(self,grad:torch.tensor,target_rank:int):
        if self.last_send is not None:
            self.last_send.wait()
        send=dist.isend(tensor=grad,dst=target_rank)
        # print("send grad from {} to {}".format(self.my_rank,target_rank))
        self.last_send=send
        return


    def receive_grad(self,source_rank:int):
        if self.last_receive is not None:
            self.last_receive.wait()
        # print("recv grad from {} to me {}".format(source_rank,self.my_rank))
        receive=dist.irecv(tensor=self.last_recv_tensor,src=source_rank)
        self.last_receive=receive

        return 



    def get_data(self):
        # Only stage0 will use this function.
        data=self.src_list[self.input_chunk_id]
        self.input_chunk_id+=1
        data=data.to(self.device)
        # print("data of pipeline ",data)
        return data

    def generate_data(self):
        # Only stage0 will use this function.
        self.input_chunk_id=0
        self.src_list=[]
        self.iteration+=1
        self.input_data[self.iteration]=[]
        '''
        # generate input data randomly
        for _ in range(self.num_chunks):
            input_tensor_shard=torch.rand(self.batch_size//self.num_chunks,self.seq_length,self.embedding_dim).requires_grad_(True).to("cuda")
            self.src_list.append(input_tensor_shard)
        '''
        for i in range(self.iteration*self.num_chunks,(self.iteration+1)*self.num_chunks):
            batch = self.train_batches[i]
            input_ids = batch['input_ids']  # Input tokens
            input_ids=input_ids.to("cuda")
            attention_mask = batch['attention_mask']  # Attention mask if needed
            self.src_list.append(input_ids)
        
        return 


    def run_pipeline(self,action_list:list):
        for action_complete in action_list:
            action_name,source_stage_id,my_stage_id,target_stage_id,chunk_id=self.parse_action(action_complete)
            
            if action_name == 'generate_data':
                self.generate_data()

            elif action_name == 'forward_first':
                self.forward_first(my_stage_id,target_stage_id,chunk_id)

            elif action_name == 'forward_middle':
                self.forward_middle(source_stage_id,my_stage_id,target_stage_id,chunk_id)

            elif action_name == 'forward_last':
                self.forward_last(source_stage_id,my_stage_id,chunk_id)

            elif action_name == 'backward_first':
                self.backward_first(my_stage_id=my_stage_id,source_stage_id=source_stage_id,chunk_id=chunk_id)
        
            elif action_name =='backward_middle':
                self.backward_middle(source_stage_id=source_stage_id,my_stage_id=my_stage_id,target_stage_id=target_stage_id,chunk_id=chunk_id)

            elif action_name =='backward_last':
                self.backward_last(my_stage_id=my_stage_id,target_stage_id=target_stage_id,chunk_id=chunk_id)

        return 


    





    
