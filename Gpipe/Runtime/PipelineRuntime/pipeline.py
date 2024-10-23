import torch
import torch.distributed as dist
from model.transformer_lm import TransformerLM
import copy
from torchviz import make_dot

def print_grad(grad):
    print("Grad:", grad)

class Pipeline():
    def __init__(self,module_list:list,world_size:int,global_rank:int,local_rank:int,batch_size:int,
    num_chunks:int,seq_length:int,embedding_dim:int,ff_dim:int,num_stages:int):
        self.module_list=module_list
        self.src_list=[] # stage0拿到的输入
        self.activation_list=[[] for _ in range(num_stages)] # 一个gpu可能负责多个stage,每个stage又有很多micro_batch的activation,所以写成长度为num_stages的数组
        self.grad_list=[[] for _ in range(num_stages)] 
        self.input_list=[[] for _ in range(num_stages)] # 虽然和activation_list中的元素是一样的东西，但是因为在my_stage也无法确定input是来自哪个stage的activation,所以单独写一个input_list
        self.batch_size=batch_size
        self.num_chunks=num_chunks
        self.seq_length=seq_length
        self.embedding_dim=embedding_dim
        self.ff_dim=ff_dim
        self.world_size=world_size
        self.num_stages=num_stages
        self.my_rank=global_rank
        self.local_rank=local_rank
        self.input_chunk_id=0
        self.last_send=None # 上一次异步发送是否成功结束的标志
        self.last_receive=None # 上一次异步接收数据是否成功的标志
        self.dtype=torch.float32
        self.total_parameters=0
        self.optimizer=self.construct_optimizer()
        self.input_data = [[] for _ in range(2)]
        self.iteration=-1
        self.device=torch.device(f'cuda:{local_rank}')
        self.last_recv_tensor=torch.zeros([self.batch_size//self.num_chunks,self.seq_length,self.embedding_dim],dtype=self.dtype,device=self.device)

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


        
 
    # 1.第一个stage需要加载数据，前向计算，向下一个stage发送计算结果[generate_data,forward_first1,f12,f13,f14]
    # 2.中间的stage需要从上个stage那里拿到计算结果，前向计算，向下一个stage发送计算结果
    # 3.最后一个stage需要从上个stage那里拿到计算结果，前向计算

    # 下面写的参数都是每个stage完整的行为

    def forward_first(self,my_stage_id:int,target_stage_id:int):
        input_data=self.get_data()
        input_data.requires_grad_(True)
        self.input_data[self.iteration].append(input_data)
        self.input_list[my_stage_id].append(input_data)
        result_tensor=self.forward_compute(input_data,my_stage_id)
        self.activation_list[my_stage_id].append(result_tensor)
        target_rank=target_stage_id%self.world_size
        self.send_activation(target_rank,result_tensor)
        # make_dot(result_tensor, params={"input_data": input_data}).render("graph", format="png")
        return 

    def forward_middle(self,source_stage_id:int,my_stage_id:int,target_stage_id:int):
        source_rank=source_stage_id%self.world_size
        # receive data
        self.receive_activation(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_data=self.last_recv_tensor.clone()
        input_data.requires_grad_(True)
        self.input_list[my_stage_id].append(input_data)
        
        result_tensor=self.forward_compute(input_data,my_stage_id)
        self.activation_list[my_stage_id].append(result_tensor)
        # send activation
        target_rank=target_stage_id%self.world_size
        self.send_activation(target_rank,result_tensor)
        return
    
    def forward_last(self,source_stage_id:int,my_stage_id:int):
        source_rank=source_stage_id%self.world_size
        # receive data
        self.receive_activation(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_data=self.last_recv_tensor.clone()
        input_data.requires_grad_(True)
        self.input_list[my_stage_id].append(input_data)
        # forward compute
        result_tensor=self.forward_compute(input_data,my_stage_id)
        self.activation_list[my_stage_id].append(result_tensor)
        # print("pipeline output ",my_stage_id,self.iteration,self.input_chunk_id,result_tensor)

        return 


    # 第一个stage需要从下一个stage接收grad,并且利用grad、本stage的activation和W计算自己的grad
    # 中间的stage需要从下一个stage接收grad,并且计算自己的grad，发送给前一个stage
    # 最后一个stage利用自己的activation计算LOSS，再计算自己的grad,发送给前一个stage

    def backward_first(self,my_stage_id:int,source_stage_id:int):
        source_rank=source_stage_id%self.world_size
        # receive data
        self.receive_grad(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_grad=self.last_recv_tensor.clone()
        # backward compute
        input_tensor=self.input_list[my_stage_id].pop()
        input_tensor.retain_grad()
        # 对于多个microbatch,后前向的先后向，符合栈的顺序
        my_activation=self.activation_list[my_stage_id].pop()
        self.backward_compute(my_activation,input_grad) 
        # print("send input grad ",my_stage_id,input_tensor.grad.shape,input_tensor.grad)
        return

    def backward_middle(self,source_stage_id:int,my_stage_id:int,target_stage_id:int):
        source_rank=source_stage_id%self.world_size
        self.receive_grad(source_rank)
        if self.last_receive is not None:
            self.last_receive.wait()
        input_grad=self.last_recv_tensor.clone()
        input_tensor=self.input_list[my_stage_id].pop()
        input_tensor.retain_grad()
        my_activation=self.activation_list[my_stage_id].pop()
        self.backward_compute(my_activation,input_grad)
        target_rank=target_stage_id%self.world_size
        self.send_grad(input_tensor.grad,target_rank)
        # if my_stage_id==1:
        #     print("send input grad ",my_stage_id,input_tensor.grad.shape,input_tensor.grad)
        return 

    def backward_last(self,my_stage_id:int,target_stage_id:int):
        # compute the gradient of input
        input_tensor=self.input_list[my_stage_id].pop()
        input_tensor.retain_grad()
        my_activation=self.activation_list[my_stage_id].pop()
        self.backward_compute(my_activation)
        target_rank=target_stage_id%self.world_size
        self.send_grad(input_tensor.grad,target_rank)

        # print("input grad ",my_stage_id,input_tensor.grad)

        return

  


    # stage完整行为中可能用到的片段操作
    def forward_compute(self,input_tensor:torch.tensor,my_stage_id:int):
        # 直接放到target_module里面去  
        module=self.module_list[my_stage_id]
        module=module.to(self.device)
        activation=module(input_tensor)
        
        return activation


    def backward_compute(self,activation:torch.tensor,accu_grad=None):
        # 按照数学公式:计算权重的梯度需要下一个stage发送来的累积+本stage的activation
        # 其实每个stage就是计算一个偏activation/偏w
        # Loss/偏w=下一个stage发送来的累积*(偏activation/偏w)
        # 这里先用.mean()来代替
        # 当backward()只有一个参数时，那个参数必须是标量；如果有grad_tensor时，activation可以是tensor类型
        if accu_grad is None:
            torch.autograd.backward(activation.mean())
        else:
            torch.autograd.backward(activation,grad_tensors=accu_grad)

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
        # 只有stage0使用
        data=self.src_list[self.input_chunk_id]
        self.input_chunk_id+=1
        data=data.to(self.device)
        # print("data of pipeline ",data)
        return data

    def generate_data(self):
        # 现在先随机生成，后续改成接入实际的模型
        # 只有stage0使用
        self.input_chunk_id=0
        self.src_list=[]
        self.iteration+=1
        self.input_data[self.iteration]=[]
        for _ in range(self.num_chunks):
            input_tensor_shard=torch.rand(self.batch_size//self.num_chunks,self.seq_length,self.embedding_dim).requires_grad_(True).to("cuda")
            self.src_list.append(input_tensor_shard)
        
        return 


    def run_pipeline(self,action_list:list):
        for action_complete in action_list:
            action_name,source_stage_id,my_stage_id,target_stage_id,chunk_id=self.parse_action(action_complete)
            
            if action_name == 'generate_data':
                self.generate_data()

            elif action_name == 'forward_first':
                self.forward_first(my_stage_id,target_stage_id)

            elif action_name == 'forward_middle':
                self.forward_middle(source_stage_id,my_stage_id,target_stage_id)

            elif action_name == 'forward_last':
                self.forward_last(source_stage_id,my_stage_id)

            elif action_name == 'backward_first':
                self.backward_first(my_stage_id=my_stage_id,source_stage_id=source_stage_id)
        
            elif action_name =='backward_middle':
                self.backward_middle(source_stage_id=source_stage_id,my_stage_id=my_stage_id,target_stage_id=target_stage_id)

            elif action_name =='backward_last':
                self.backward_last(my_stage_id=my_stage_id,target_stage_id=target_stage_id)
   
            # print(self.my_rank,action_complete)
        return 


    





    
