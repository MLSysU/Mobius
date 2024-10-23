def generate_action_list(world_size:int,num_stages:int,num_chunks:int):
    # 生成每个GPU要执行的动作
    action_list=[ [] for _ in range(world_size)]
    action_list[0].append("generate_data -1 0 -1 -1") # 第一个stage要负责生成输入数据

    for stage_id in range(num_stages):
        last_stage_id=stage_id-1
        next_stage_id=stage_id+1

        # generate forward process
        if stage_id==0:
            for chunk_id in range(num_chunks):
                action_list[0].append("forward_first -1 0 {} {}".format(next_stage_id,chunk_id))
        elif stage_id==num_stages-1:
            for chunk_id in range(num_chunks):
                action_list[stage_id%world_size].append("forward_last {} {} -1 {}".format(last_stage_id,stage_id,chunk_id))             

        else:
            for chunk_id in range(num_chunks):
                action_list[stage_id%world_size].append("forward_middle {} {} {} {}".format(last_stage_id,stage_id,next_stage_id,chunk_id))

    for stage_id in range(num_stages-1,-1,-1):
        last_stage_id=stage_id-1
        next_stage_id=stage_id+1    
        # generate backward process
        if stage_id==0:
            for chunk_id in range(num_chunks):
                action_list[0].append("backward_first {} 0 -1 {}".format(next_stage_id,chunk_id))

        elif stage_id==num_stages-1:
            for chunk_id in range(num_chunks):
                action_list[stage_id%world_size].append("backward_last -1 {} {} {}".format(stage_id,last_stage_id,chunk_id))

        else:
            for chunk_id in range(num_chunks):
                action_list[stage_id%world_size].append("backward_middle {} {} {} {}".format(next_stage_id,stage_id,last_stage_id,chunk_id))
    
    return action_list