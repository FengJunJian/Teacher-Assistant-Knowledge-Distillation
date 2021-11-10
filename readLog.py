import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json
import os
#dalu=open("处理后.log","w+",encoding='UTF-8')
font=font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

root='log'
summary=['8-8','56-8','110-8']#  os.listdir(root)
epoch = 50
legend1=[]
legend2=[]
ax1=plt.figure(1).add_subplot()
ax2=plt.figure(2).add_subplot()
for k in range(len(summary)):
    with open(os.path.join(root,summary[k],"trial.log"),"r",encoding='UTF-8') as f:
        #log=json.load(f)
        log=f.readlines()

    index=0
    Tvalues=[]
    Svalues=[]
    for i,j in enumerate(range(4,len(log),2)):
        if i>=epoch:
            index = j
            break
        print(i,j)
        K_V=json.loads(log[j][log[j].index("{"):].strip())

        Tvalues.append(K_V['value'])

    for i,j in enumerate(range(index+1,len(log),2)):
        if i>=epoch:
            break
        print(i, j)
        K_V = json.loads(log[j][log[j].index("{"):].strip())
        Svalues.append(K_V['value'])


    ax1.plot(Tvalues)
    ax1.plot(Svalues)
    #ax2=plt.figure(2)
    ax2.plot(Svalues)
    legend1+=[summary[k]+'教师',summary[k]+'学生']
    legend2 += [ summary[k] + '学生']
ax1.legend(legend1)
ax1.set_title('分类精度')
ax1.set_xlabel('轮次')
ax1.set_ylabel('精度(%)')
ax1.grid()

ax2.legend(legend2)
ax2.set_title('分类精度')
ax2.set_xlabel('轮次')
ax2.set_ylabel('精度(%)')
ax2.grid()
plt.show()
    # file.close()

