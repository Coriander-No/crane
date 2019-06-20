import math
import threading
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

global L0
global L1 #工位1
global L2 #工位2
global L3 #工位3
global L4 #工位4
global L5
global safeL # 安全距离
global t_up #起吊时间
global t_down #卸载时间
global v #速度
global left
global right
L0 = 0
L1 = 10
L2 = 110
L3 = 210
L4 = 310
L5 = 320
safeL = 10
t_up = 1
t_down = 1
v = 10
left = 0
right = 1


# 行车模型
class Crane(object):
    def __init__(self, locate, cargo, priority):
        self.locate = locate #位置
        self.cargo = cargo #货物的有无
        self.priority = priority # 优先级 -1(空车等待) 0(空车运行) 1(载重行驶) 2(起吊/卸载)

    # 装载货物    
    def set_Cargo(self):
        self.cargo = 1

    # 卸载货物
    def put_Cargo(self):
        self.cargo = 0

    # 左移
    def move_Lelf(self):
        self.locate -= v

    # 右移    
    def move_Right(self):
        self.locate += v
    
    # 更改优先级
    def change_priority(self, i):
        self.priority = i

    # 读取位置
    def get_Locate(self):
        return self.locate

    # 读取行车状态(是否装载货物)
    def get_Cargo(self):
        return self.cargo

    # 读取当前优先级
    def get_Priority(self):
        return self.priority


#多线程
class myThread(threading.Thread):
    def __init__(self, func, args=()):
        super(myThread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
   

def main():
    pop_size = 10 #种群数量
    chromosome_length = 10 #染色体长度
    iter = 100 #遗传代数
    pc = 0.5 # 杂交概率
    pm = 0.01 # 变异概率
    results = [] # 存储每代最优解
    pop = init_population(pop_size, chromosome_length)
    p = [Crane(10,0,-1),Crane(110, 0, -1),Crane(210,0,-1)]
    i = 0
    while i < iter:
        obj_value = pop_Move(pop,p,pop_size,chromosome_length) #计算结果
        fit_value = calc_fit_value(pop, pop_size, obj_value) # 淘汰 
        #每代最佳的结果
        best_individual, best_fit = find_best(pop, fit_value)
        if best_individual:
            results.append([binary2decimal(best_individual),best_fit])
            selection(pop, fit_value)
            crossover(pop, pc)
            mutation(pop, pm)
            i += 1
        else:
            selection(pop, fit_value)
            crossover(pop, pc)
            mutation(pop, pm)
    plot_iter_curve(iter, results)
    m = results[0][1]
    best = results[0][0]
    for i in range(100):
        if results[i][1] < m:
            m = results[i][1]
            best = results[i][0]
    print(m)
    print(best)
    best_result = dec2bin(best)
    print(best_result)
    best_result_crane = best_move(p,best_result)
    plot_move_crane(best_result_crane)


        

def moveTask(p, num, direct):
    if direct == right:
        if p[num].get_Cargo() == 0:
            p[num].move_Right()
            p[num].change_priority(0)
        if p[num].get_Cargo() == 1:
            p[num].move_Right()
            p[num].change_priority(1)
    if direct == left:
        if p[num].get_Cargo() == 0:
            p[num].move_Lelf()
            p[num].change_priority(0)
        if p[num].get_Cargo() == 1:
            p[num].move_Lelf()
            p[num].change_priority(1) 
    return p[num]

#初始化种群
def init_population(pop_size, chromosome_length):
    pop = [ j for j in range(pop_size)]
    m = 0
    while m < chromosome_length:
        pop[m] = [random.randint(0,2) for i in range(10)]
        if pop[m].count(0) >2  and pop[m].count(1) > 2 and pop[m].count(2) > 2 :
            m += 1
    return pop
    
def pop_Move(pop,p,pop_size,chromosome_length):
    thread = []
    t1 = myThread(moveTask,(p,0,0))
    t2 = myThread(moveTask,(p,1,0))
    t3 = myThread(moveTask,(p,2,0))
    thread.append(t1)
    thread.append(t2)
    thread.append(t3)
    obj_value = []
    for m in range(pop_size):
        p = [Crane(10, 0, -1),Crane(110, 0, -1),Crane(210,0,-1)]
        time = 0 
        for i in range(chromosome_length):
            if pop[m][i] == 0:
                while p[0].get_Locate() > L1 and p[0].get_Locate() < L2 and p[0].get_Priority() <= 0:
                    thread[0] = myThread(moveTask,(p,0,left))
                    thread[0].start()
                    thread[0].join()
                    p[0] = thread[0].get_result()
                    time += 1


                while p[0].get_Locate() > L2 and p[0].get_Priority() <= 0:
                    thread[0] = myThread(moveTask,(p,0,left))
                    thread[0].start()
                    thread[0].join()
                    p[0] = thread[0].get_result()
                    time += 1
                   

                if (p[0].get_Locate() == L1 or p[0].get_Locate() == L2) and p[0].get_Priority() <= 0:
                    p[0].set_Cargo()
                    p[0].change_priority(2)
                    time += t_up
                    if p[0].get_Locate() == L2:
                        thread[0] = myThread(moveTask,(p,0,right))
                        if p[0].get_Locate() + safeL >= p[1].get_Locate():
                            thread[1] = myThread(moveTask,(p,1,right))
                            thread[0].start()
                            thread[1].start()
                            thread[0].join()
                            thread[1].join()
                            p[0] = thread[0].get_result()
                            p[1] = thread[1].get_result()
                        else:
                            thread[0].start()
                            thread[0].join()
                            p[0] = thread[0].get_result()
                        time += 1

                        
                while p[0].get_Locate() < L2 and p[0].get_Priority() > 0:
                    thread[0] = myThread(moveTask,(p, 0, right))
                    if p[0].get_Locate() + safeL >= p[1].get_Locate():
                        thread[1] = myThread(moveTask,(p, 1, right))
                        thread[0].start()
                        thread[1].start()
                        thread[0].join()
                        thread[1].join() 
                        p[0] = thread[0].get_result()
                        p[1] = thread[1].get_result() 
                    else:
                        thread[0].start()
                        thread[0].join()
                        p[0] = thread[0].get_result() 
                    time += 1


                while p[0].get_Locate() < L3 and p[0].get_Locate() > L2 and p[0].get_Priority() > 0:
                    thread[0] = myThread(moveTask,(p, 0, right))
                    if p[0].get_Locate() + safeL >= p[1].get_Locate():
                        thread[1] = myThread(moveTask,(p, 1, right))
                        if p[1].get_Locate() + safeL >= p[2].get_Locate():
                            thread[2] = myThread(moveTask,(p,2,right))
                            thread[0].start()
                            thread[1].start()
                            thread[2].start()
                            thread[0].join()
                            thread[1].join() 
                            thread[2].join()
                            p[0] = thread[0].get_result()
                            p[1] = thread[1].get_result()
                            p[2] = thread[2].get_result()
                        else: 
                            thread[0].start()
                            thread[1].start()
                            thread[0].join()
                            thread[1].join() 
                            p[0] = thread[0].get_result()
                            p[1] = thread[1].get_result() 
                    else:
                        thread[0].start()
                        thread[0].join()
                        p[0] = thread[0].get_result() 
                    time += 1


                if (p[0].get_Locate() == L2 or p[0].get_Locate() == L3) and p[0].get_Cargo() == 1:
                    p[0].put_Cargo()
                    p[0].change_priority(2)
                    time += t_down

                p[0].change_priority(-1)  
    
        # 二号 车
            elif pop[m][i] == 1:
                while p[1].get_Locate() > L2 and p[1].get_Locate() < L3 and p[1].get_Priority() <= 0:
                    thread[1] = myThread(moveTask, (p, 1, left))
                    if p[0].get_Locate() + safeL >= p[1].get_Locate():
                        thread[0] = myThread(moveTask,(p, 0, left))
                        thread[0].start()
                        thread[1].start()
                        thread[0].join()
                        thread[1].join() 
                        p[0] = thread[0].get_result()
                        p[1] = thread[1].get_result() 
                    else:
                        thread[1].start()
                        thread[1].join()
                        p[1] = thread[1].get_result()
                    time += 1

                while p[1].get_Locate() > L3 and p[1].get_Priority() <= 0:
                    thread[1] = myThread(moveTask, (p, 1, left))
                    if p[0].get_Locate() + safeL >= p[1].get_Locate():
                        thread[0] = myThread(moveTask,(p, 0, left))
                        thread[0].start()
                        thread[1].start()
                        thread[0].join()
                        thread[1].join() 
                        p[0] = thread[0].get_result()
                        p[1] = thread[1].get_result() 
                    else:
                        thread[1].start()
                        thread[1].join()
                        p[1] = thread[1].get_result()
                    time += 1


                if p[1].get_Locate() == L2 or p[1].get_Locate() == L3 and p[1].get_Priority() <= 0:
                    p[1].set_Cargo()
                    p[1].change_priority(2)
                    time += t_up
                    thread[1] = myThread(moveTask,(p,1,right))
                    if p[1].get_Locate() + safeL >= p[2].get_Locate():
                        thread[2] = myThread(moveTask,(p,2,right))
                        thread[1].start()
                        thread[2].start()
                        thread[1].join()
                        thread[2].join()
                        p[1] = thread[1].get_result()
                        p[2] = thread[2].get_result()
                    else:
                        thread[1].start()
                        thread[1].join()
                        p[1] = thread[1].get_result()
                    time += 1

                
                while p[1].get_Locate() < L3 and p[1].get_Locate() > L2 and p[1].get_Priority() > 0:
                    thread[1] = myThread(moveTask,(p, 1, right))
                    if p[1].get_Locate() + safeL >= p[2].get_Locate():
                        thread[2] = myThread(moveTask,(p,2,right))
                        thread[1].start()
                        thread[2].start()
                        thread[1].join()
                        thread[2].join()
                        p[1] = thread[1].get_result()
                        p[2] = thread[2].get_result()
                    else:
                        thread[1].start()
                        thread[1].join()
                        p[1] = thread[1].get_result() 
                    time += 1

                while p[1].get_Locate() > L3 and p[1].get_Locate() < L4 and p[1].get_Priority() > 0:
                    thread[1] = myThread(moveTask,(p, 1, right))
                    if p[1].get_Locate() + safeL >= p[2].get_Locate():
                        thread[2] = myThread(moveTask,(p,2,right))
                        thread[1].start()
                        thread[2].start()
                        thread[1].join()
                        thread[2].join()
                        p[1] = thread[1].get_result()
                        p[2] = thread[2].get_result()
                    else:
                        thread[1].start()
                        thread[1].join()
                        p[1] = thread[1].get_result() 
                    time += 1
                    

                if (p[1].get_Locate() == L3 or p[1].get_Locate() == L4) and p[1].get_Cargo() == 1:
                    p[1].put_Cargo()
                    p[1].change_priority(2)
                    time += t_down
                    
                p[1].change_priority(-1) 
        # 三车
            else:
                while p[2].get_Locate() > L3 and p[2].get_Priority() <= 0:
                    thread[2] = myThread(moveTask,(p,2,left))
                    if p[1].get_Locate() + safeL >= p[2].get_Locate():
                        thread[1] = myThread(moveTask,(p,1,left))
                        if p[0].get_Locate() + safeL >= p[1].get_Locate():
                            thread[0] = myThread(moveTask,(p,0,left))
                            thread[0].start()
                            thread[1].start()
                            thread[2].start()
                            thread[0].join()
                            thread[1].join()
                            thread[2].join()
                            p[0] = thread[0].get_result()
                            p[1] = thread[1].get_result()
                            p[2] = thread[2].get_result()
                        else:
                            thread[1].start()
                            thread[2].start()
                            thread[1].join()
                            thread[2].join()
                            p[1] = thread[1].get_result()
                            p[2] = thread[2].get_result()
                    else:
                        thread[2].start()
                        thread[2].join()
                        p[2] = thread[2].get_result()
                    time += 1
                if p[2].get_Locate() == L3 and p[2].get_Priority() <= 0:
                    p[2].set_Cargo()
                    p[2].change_priority(2)
                    time += t_up

                while p[2].get_Locate() < L4 and p[2].get_Priority() > 0:
                    thread[2] = myThread(moveTask,(p,2,right))
                    thread[2].start()
                    thread[2].join()
                    p[2] = thread[2].get_result()
                    time += 1

                if p[2].get_Locate() == L4 and p[2].get_Priority() > 0:
                    p[2].put_Cargo()
                    p[2].change_priority(2)
                    time += t_down
                p[2].change_priority(-1)
        obj_value.append(time)                                             
    return obj_value 
   
#最终迭代曲线
def plot_iter_curve(iter, results):
    X = [i for i in range(iter)]
    Y = [results[i][1] for i in range(iter)]
    for i in range(iter):
        print('{num}    {task}, {time}'.format(num = i+1, task = results[i][0], time = results[i][1]))
    plt.plot(X, Y)
    plt.xlabel('generation',fontsize=25)
    plt.ylabel('time',fontsize=25)
    plt.show()

#淘汰
def calc_fit_value(pop, pop_size, obj_value):
    fit_value = []
    max = 0    
    for i in obj_value:
        max += i
    c_max = max / 10
    for j in range(pop_size):
        if pop[j].count(0) < 2 or pop[j].count(1) < 2 or pop[j].count(2) < 2:
            obj_value[j] = 0   
    for value in obj_value:
        if value < c_max:
            temp = value
        else:
            temp = 0
        fit_value.append(temp)
    return fit_value

#最优解
def find_best(pop, fit_value):
    best_individual = []
    for i in range(len(pop)):
        if fit_value[i] != 0  :
            best_fit = fit_value[i]
            break
    for j in range(1, len(pop)):
        if fit_value[j] < best_fit  and fit_value[j] != 0:
            best_fit = fit_value[j]
            best_individual = pop[j]

    return best_individual, best_fit

#计算2进制序列代表的数值
def binary2decimal(binary):
    t = 0
    for j in range(len(binary)):
        t += binary[j] * 3 ** j  
    return t

def cum_sum(fit_value):
    temp = fit_value[:]
    for i in range(len(temp)):
        fit_value[i] = (sum(temp[:i + 1]))

#轮盘赌法选择
def selection(pop, fit_value):
    print(fit_value)
    p_fit_value = []
    anti_fit_value = []
    for i in range(10):
        if fit_value[i] != 0:
            anti_fit_value.append(180 - fit_value[i])
        else:
            anti_fit_value.append(fit_value[i])
    print(anti_fit_value)
    # 适应度总和
    total_fit = sum(anti_fit_value)
    for i in range(len(anti_fit_value)):
        p_fit_value.append(anti_fit_value[i] / total_fit)
    # 计算累计概率
    cum_sum(p_fit_value)
    pop_len = len(pop)
    ms = sorted([random.random() for i in range(pop_len)])
    fitin = 0
    newin = 0
    newpop = pop[:]
    # 轮盘赌选择法
    while newin < pop_len:
        if(ms[newin] < p_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
        pop = newpop[:]
    random.shuffle(pop)
    print(pop)

#杂交
def crossover(pop, pc):
    #杂交种群相邻的两个个体
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if(random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1[:]
            pop[i + 1] = temp2[:]

# 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if(random.random() < pm):
            mpoint = random.randint(0, py -1)
            if(pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1

def dec2bin(best):
    best_Task = []
    i = 0
    while best > 0:
        bi = best % 3
        best = int(best / 3)
        best_Task.append(bi)
        i += 1
    while i < 10:
        best_Task.append(0)
        i += 1
    return best_Task

def best_move(p,best_result):
    thread = []
    t1 = myThread(moveTask,(p,0,0))
    t2 = myThread(moveTask,(p,1,0))
    t3 = myThread(moveTask,(p,2,0))
    thread.append(t1)
    thread.append(t2)
    thread.append(t3)
    result_crane = []
    time = 0
    p = [Crane(10, 0, -1),Crane(110, 0, -1),Crane(210,0,-1)]
    result_crane.append([10,110,210])
    for i in range(10):
        if best_result[i] == 0:
            while p[0].get_Locate() > L1 and p[0].get_Locate() < L2 and p[0].get_Priority() <= 0:
                thread[0] = myThread(moveTask,(p,0,left))
                thread[0].start()
                thread[0].join()
                p[0] = thread[0].get_result()
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 

            while p[0].get_Locate() > L2 and p[0].get_Priority() <= 0:
                thread[0] = myThread(moveTask,(p,0,left))
                thread[0].start()
                thread[0].join()
                p[0] = thread[0].get_result()
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()])                     

            if (p[0].get_Locate() == L1 or p[0].get_Locate() == L2) and p[0].get_Priority() <= 0:
                p[0].set_Cargo()
                p[0].change_priority(2)
                time += t_up
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
                if p[0].get_Locate() == L2:
                    thread[0] = myThread(moveTask,(p,0,right))
                    if p[0].get_Locate() + safeL >= p[1].get_Locate():
                        thread[1] = myThread(moveTask,(p,1,right))
                        thread[0].start()
                        thread[1].start()
                        thread[0].join()
                        thread[1].join()
                        p[0] = thread[0].get_result()
                        p[1] = thread[1].get_result()
                    else:
                        thread[0].start()
                        thread[0].join()
                        p[0] = thread[0].get_result()
                    time += 1
                    result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
                    
            while p[0].get_Locate() < L2 and p[0].get_Priority() > 0:
                thread[0] = myThread(moveTask,(p, 0, right))
                if p[0].get_Locate() + safeL >= p[1].get_Locate():
                    thread[1] = myThread(moveTask,(p, 1, right))
                    thread[0].start()
                    thread[1].start()
                    thread[0].join()
                    thread[1].join() 
                    p[0] = thread[0].get_result()
                    p[1] = thread[1].get_result() 
                else:
                    thread[0].start()
                    thread[0].join()
                    p[0] = thread[0].get_result() 
                time += 1 
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()])  
            while p[0].get_Locate() < L3 and p[0].get_Locate() > L2 and p[0].get_Priority() > 0:
                thread[0] = myThread(moveTask,(p, 0, right))
                if p[0].get_Locate() + safeL >= p[1].get_Locate():
                    thread[1] = myThread(moveTask,(p, 1, right))
                    if p[1].get_Locate() + safeL >= p[2].get_Locate():
                        thread[2] = myThread(moveTask,(p,2,right))
                        thread[0].start()
                        thread[1].start()
                        thread[2].start()
                        thread[0].join()
                        thread[1].join() 
                        thread[2].join()
                        p[0] = thread[0].get_result()
                        p[1] = thread[1].get_result()
                        p[2] = thread[2].get_result()
                    else: 
                        thread[1] = myThread(moveTask,(p, 1, right))
                        thread[0].start()
                        thread[1].start()
                        thread[0].join()
                        thread[1].join() 
                        p[0] = thread[0].get_result()
                        p[1] = thread[1].get_result() 
                else:
                    thread[0].start()
                    thread[0].join()
                    p[0] = thread[0].get_result() 
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()])  
            if (p[0].get_Locate() == L2 or p[0].get_Locate() == L3) and p[0].get_Cargo() == 1:
                p[0].put_Cargo()
                p[0].change_priority(2)
                time += t_down
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
            p[0].change_priority(-1)  

    # 二号 车
        elif best_result[i] == 1:
            while p[1].get_Locate() > L2 and p[1].get_Locate() < L3 and p[1].get_Priority() <= 0:
                thread[1] = myThread(moveTask, (p, 1, left))
                if p[0].get_Locate() + safeL >= p[1].get_Locate():
                    thread[0] = myThread(moveTask,(p, 0, left))
                    thread[0].start()
                    thread[1].start()
                    thread[0].join()
                    thread[1].join() 
                    p[0] = thread[0].get_result()
                    p[1] = thread[1].get_result() 
                else:
                    thread[1].start()
                    thread[1].join()
                    p[1] = thread[1].get_result()
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
            while p[1].get_Locate() > L3 and p[1].get_Priority() <= 0:
                thread[1] = myThread(moveTask, (p, 1, left))
                if p[0].get_Locate() + safeL >= p[1].get_Locate():
                    thread[0] = myThread(moveTask,(p, 0, left))
                    thread[0].start()
                    thread[1].start()
                    thread[0].join()
                    thread[1].join() 
                    p[0] = thread[0].get_result()
                    p[1] = thread[1].get_result() 
                else:
                    thread[1].start()
                    thread[1].join()
                    p[1] = thread[1].get_result()
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 

            if p[1].get_Locate() == L2 or p[1].get_Locate() == L3 and p[1].get_Priority() <= 0:
                p[1].set_Cargo()
                p[1].change_priority(2)
                time += t_up
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
                thread[1] = myThread(moveTask,(p,1,right))
                if p[1].get_Locate() + safeL >= p[2].get_Locate():
                    thread[2] = myThread(moveTask,(p,2,right))
                    thread[1].start()
                    thread[2].start()
                    thread[1].join()
                    thread[2].join()
                    p[1] = thread[1].get_result()
                    p[2] = thread[2].get_result()
                else:
                    thread[1].start()
                    thread[1].join()
                    p[1] = thread[1].get_result()
                    time += 1
                    result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
            while p[1].get_Locate() < L3 and p[1].get_Locate() > L2 and p[1].get_Priority() > 0:
                thread[1] = myThread(moveTask,(p, 1, right))
                if p[1].get_Locate() + safeL >= p[2].get_Locate():
                    thread[2] = myThread(moveTask,(p,2,right))
                    thread[1].start()
                    thread[2].start()
                    thread[1].join()
                    thread[2].join()
                    p[1] = thread[1].get_result()
                    p[2] = thread[2].get_result()
                else:
                    thread[1].start()
                    thread[1].join()
                    p[1] = thread[1].get_result() 
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
            while p[1].get_Locate() > L3 and p[1].get_Locate() < L4 and p[1].get_Priority() > 0:
                thread[1] = myThread(moveTask,(p, 1, right))
                if p[1].get_Locate() + safeL >= p[2].get_Locate():
                    thread[2] = myThread(moveTask,(p,2,right))
                    thread[1].start()
                    thread[2].start()
                    thread[1].join()
                    thread[2].join()
                    p[1] = thread[1].get_result()
                    p[2] = thread[2].get_result()
                else:
                    thread[1].start()
                    thread[1].join()
                    p[1] = thread[1].get_result() 
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()])                       
            if (p[1].get_Locate() == L3 or p[1].get_Locate() == L4) and p[1].get_Cargo() == 1:
                p[1].put_Cargo()
                p[1].change_priority(2)
                time += t_down
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
                
            p[1].change_priority(-1) 
        else:
            while p[2].get_Locate() > L3 and p[2].get_Priority() <= 0:
                thread[2] = myThread(moveTask,(p,2,left))
                if p[1].get_Locate() + safeL >= p[2].get_Locate():
                    thread[1] = myThread(moveTask,(p,1,left))
                    if p[0].get_Locate() + safeL >= p[1].get_Locate():
                        thread[0] = myThread(moveTask,(p,0,left))
                        thread[0].start()
                        thread[1].start()
                        thread[2].start()
                        thread[0].join()
                        thread[1].join()
                        thread[2].join()
                        p[0] = thread[0].get_result()
                        p[1] = thread[1].get_result()
                        p[2] = thread[2].get_result()
                    else:
                        thread[1].start()
                        thread[2].start()
                        thread[1].join()
                        thread[2].join()
                        p[1] = thread[1].get_result()
                        p[2] = thread[2].get_result()
                else:
                    thread[2].start()
                    thread[2].join()
                    p[2] = thread[2].get_result()
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
            if p[2].get_Locate() == L3 and p[2].get_Priority() <= 0:
                p[2].set_Cargo()
                p[2].change_priority(2)
                time += t_up
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
            while p[2].get_Locate() < L4 and p[2].get_Priority() > 0:
                thread[2] = myThread(moveTask,(p,2,right))
                thread[2].start()
                thread[2].join()
                p[2] = thread[2].get_result()
                time += 1
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()]) 
            if p[2].get_Locate() == L4 and p[2].get_Priority() > 0:
                p[2].put_Cargo()
                p[2].change_priority(2)
                time += t_down
                result_crane.append([p[0].get_Locate(), p[1].get_Locate(), p[2].get_Locate()])
            p[2].change_priority(-1)
                                                                                                            
    return result_crane 

def plot_move_crane(result_crane):
    length = len(result_crane)
    X = [i for i in range(length)]
    Y1 = [result_crane[i][0] for i in range(length)]
    Y2 = [result_crane[i][1] for i in range(length)]
    Y3 = [result_crane[i][2] for i in range(length)] 
    my_x_ticks = np.arange(0, 180, 5)
    my_y_ticks = np.arange(0, 420, 10)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.plot(X,Y1,color = 'green', label = 'crane 1')
    plt.plot(X,Y2,color = 'red', label = 'crane 2',linestyle = '-.')
    plt.plot(X,Y3,color = 'black', label = 'crane 3', linestyle = '--')
    plt.xlabel('time',fontsize=25)
    plt.ylabel('locate',fontsize=25)    
    plt.legend()
    plt.show()


 
if __name__ == '__main__':
    main()
