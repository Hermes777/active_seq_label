import random
_input="ibm.txt"
label_input=_input+'.label.txt'
string_input=_input+'.string.txt'
fi1=open(label_input,"r")
fi2=open(string_input,"r")
fo1_=open(_input[:-4]+'.train'+'.label.txt',"w+")
fo2_=open(_input[:-4]+'.train'+'.string.txt',"w+")
fo1__=open(_input[:-4]+'.dev'+'.label.txt',"w+")
fo2__=open(_input[:-4]+'.dev'+'.string.txt',"w+")
fo1=open(_input[:-4]+'.test'+'.label.txt',"w+")
fo2=open(_input[:-4]+'.test'+'.string.txt',"w+")

s=[]
l=[]
for x in fi1:
    l.append(x)
for x in fi2:
    s.append(x)

fi3=range(len(l))
print("le",len(fi3[:-400]))
random.seed(1)
random.shuffle(fi3)

for x in fi3[:-400]:
    y=l[x]
    print >>fo1_,y[:-1]
    y=s[x]
    print >>fo2_,y[:-1]

for x in fi3[-400:-200]:
    y=l[x]
    print >>fo1__,y[:-1]
    y=s[x]
    print >>fo2__,y[:-1]

for x in fi3[-200:]:
    y=l[x]
    print >>fo1,y[:-1]
    y=s[x]
    print >>fo2,y[:-1]
