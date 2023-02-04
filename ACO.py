from random import seed
from random import randint
import array
import numpy as np

#Function 1
def five_uneven_peak_trap(x):
    
    if (x>=0 and x<2.50):
        result = 80*(2.5-x)
    elif (x>=2.5 and x<5):
        result = 64*(x-2.5)
    elif (x >= 5.0 and x < 7.5):
        result = 64*(7.5-x)
    elif (x >= 7.5 and x < 12.5):
        result = 28*(x-7.5)
    elif (x >= 12.5 and x < 17.5):
        result = 28*(17.5-x)
    elif (x >= 17.5 and x < 22.5):
        result = 32*(x-17.5)
    elif (x >= 22.5 and x < 27.5):
        result = 32*(27.5-x)
    elif (x >= 27.5 and x <= 30):
        result = 80*(x-27.5)
    elif (x<0 or x>30):
        result = -1
    return result
    
#Function 2    
def equal_maxima(x):
    return np.sin(5.0 * np.pi * x)**6
    
#Function 3
def uneven_decreasing_maxima(x):
	return np.exp(-2.0*np.log(2)*((x-0.08)/0.854)**2)*(np.sin(5*np.pi*(x**0.75-0.05)))**6
	
#Function 4
def himmelblau(x):
	result = 200 - (x[0]**2 + x[1] - 11)**2 - (x[0] + x[1]**2 - 7)**2
	return result
	
#Function 5
def six_hump_camel_back(x):
	x2 = x[0]**2
	x4 = x[0]**4
	y2 = x[1]**2
	expr1 = (4.0 - 2.1*x2 + x4/3.0)*x2
	expr2 = x[0]*x[1]
	expr3 = (4.0*y2 - 4.0)*y2
	return -1.0*(expr1+expr2+expr3)
	
#Function 6
def shubert(x):
	i = 0
	result = 1
	soma = [0]*len(x)
	D = len(x)


	while i < D:
		for j in range (1, 6):
			soma[i] = soma[i] + (j*math.cos((j+1)*x[i]+j))
		result = result*soma[i]
		i = i + 1
	return -result

#Function 7
def vincent(x):
	result = 0
	D = len(x)

	for i in range(0, D):
		result += (math.sin(10*math.log(x[i])))/D
	return result

#Algorithm 8
def modified_rastrigin_all(x):
	result = 0
	D = len(x)    
	if D==2:
		k = [3, 4]
	elif D==8:
		k = [1, 2, 1, 2, 1, 3, 1, 4]
	elif D==16:
		k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]

	for i in range (0, D):
		result += (10 + 9*math.cos(2*math.pi*k[i]*x[i]))        
	return -result
	
	
#Algorithm 1
def clustering_for_crowding(a,Clustersize):
    #print(Clustersize)
   # print(a)
    r=randint(0,len(a)-1)
    #print(r)
    sorted_arr = np.empty(len(a))
    n = np.empty(len(a))
    for i in range(0,len(a)-1):
            x = (r-i)*0.5
            if(x<0):
               x=x*(-1)
            n[i]=x
    flag=0
    
    #for function 1,2,3
    fl,j,l,k=0,0,1,r
    while flag==0:
      for i in range(0,Clustersize-1):
          if(j==0):
             sorted_arr[j]=a[k]
             a[k],j=-10,1
          elif(k+l<80 and k-l>-1):
               if(fl==0):
                  sorted_arr[j]=a[k-l]
                  a[k-l],j,fl=-10,j+1,1
               elif(fl==1):
                  sorted_arr[j]=a[k+l]
                  a[k+l]=-10
                  l,j,fl=l+1,j+1,0
          elif(k+l<80):
               sorted_arr[j]=a[k+l]
               a[k+l]=-10
               l+=1
               j+=1
          elif(k-l>-1):
               sorted_arr[j]=a[k-l]
               a[k-l],l,j=-10,l+1,j+1
      flag=1
      for i in range(0,len(a)-1):
          if(a[i]!=-10):
             flag=0     
    #print(sorted_arr)
    #print(a)
    return sorted_arr
    
#Algorithm 2
def clustering_for_speciation(a,Clustersize):
    sorted_arr=np.empty(len(a))
    ranked_arr= np.empty(len(a))
    for i in range(0,len(a)):
        rank=79
        for m in range(0,len(a)):
            if(five_uneven_peak_trap(a[m])<five_uneven_peak_trap(a[i])):
               rank-=1
        #print(i)
        #print(rank)
        sorted_arr[rank]=a[i]
    return sorted_arr
    
#Algorithm 4
def solution_contruction(Clustersize,sorted_arr,f_max,f_min):
    new_soloutions_arr = np.empty(len(sorted_arr))
    probabilities_arr = np.empty(len(sorted_arr))
    if(len(sorted_arr)%Clustersize==0):
       num_of_clusters=(len(sorted_arr)/Clustersize)
    elif(len(sorted_arr)%Clustersize!=0):
       num_of_clusters=int(round(len(sorted_arr)/Clustersize)+1)
    n=int(num_of_clusters)
   # print(type(n))
    k=0
    for i in range(0,n-1):
        n_max=sorted_arr[k]
        n_min=sorted_arr[k]
        for q in range(k,(k+Clustersize)-1):
            if(sorted_arr[q]>n_max):
               n_max=sorted_arr[q]
            elif(sorted_arr[q]<n_min):
                 n_min=sorted_arr[q]
        sd=0.1 + 0.3 * (2.718 ** ((n_max-n_min)/(f_max-f_min+0.1)))
        
        sum_weights=0
        for h in range(1,len(sorted_arr)):
            rank_i=79
            for m in range(0,len(sorted_arr)):
                if(five_uneven_peak_trap(sorted_arr[m])<five_uneven_peak_trap(sorted_arr[h])):
                   rank_i-=1
            weight_i=1 / (sd*len(sorted_arr)*(((2 * 3.142)**0.5)**( (2.718)**(((rank_i-1)**2)/(2*(sd**2)*((len(sorted_arr))**2))))))
            sum_weights+=weight_i 
        
        for q in range(k,(k+Clustersize)-1):
            rank=79
            for m in range(0,len(sorted_arr)):
                if(five_uneven_peak_trap(sorted_arr[m])<five_uneven_peak_trap(sorted_arr[q])):
                   rank-=1
            weight=1 / (sd*len(sorted_arr)*(((2 * 3.142)**0.5)**( (2.718)**(((rank-1)**2)/(2*(sd**2)*((len(sorted_arr))**2))))))
        
            probabilities_arr[q]=weight / sum_weights
            
        for q in range(k,(k+Clustersize)-1):
            d=randint(k,(k+Clustersize)-1)
            if(five_uneven_peak_trap(sorted_arr[d])<=0.5):
               mu=sorted_arr[d]
            elif(five_uneven_peak_trap(sorted_arr[d])>0.5):
                 fr=randint(0,1)
                 mu=sorted_arr[d]+fr
            a=0
            for g in range(1,len(sorted_arr)):
                b=sorted_arr[g]-sorted_arr[q]
                if(b<0):
                    b*=-1
                a+= b/(len(sorted_arr)-1)
            
            ri=randint(0,1)
            while ri==0:
                ri=randint(0,1)
            new_soloutions_arr[q]= 1 / ((ri*a) * (((2*3.142)**2.718)**((((sorted_arr[q]-mu)**2)/(2*(((ri*a))**2)))*-1)))
        k+=Clustersize

    return new_soloutions_arr

#Algorithm 5
def adaptive_local_search(final_arr,le,c,sa,N):
    prob=np.empty(len(final_arr))
    f_max=c[0];
    f_min=c[0];
    for i in range(0,len(final_arr)-1):
        if(c[i]>f_max):
            f_max=c[i]
        elif(c[i]<f_min):
            f_min=c[i]
            
    flag=False
    if(f_min<=0):
        if(f_min<0):
            t=f_min*(-1)
        elif(f_min==0):
            t=f_min
        f_max,flag=f_max+t+0.1,True
    for i in range(0,le-1):
        if(flag==True):
            prob[i]=((five_uneven_peak_trap(final_arr[i])+t+0.1)/ (f_max+t+0.1))
        elif(flag==False):
            prob[i]=five_uneven_peak_trap(final_arr[i])/f_max
    for i in range(0,le-1):
        num=randint(0,1)
        while num==0:
            num=randint(0,1)
        if(num<=prob[i]):
            for j in  range(0,N-1):
                new_solution=1 / (sa * (((2*3.142)**2.718)**((((final_arr[i]-final_arr[i])**2)/(2*((sa)**2)))*-1)))
                if(five_uneven_peak_trap(new_solution)>five_uneven_peak_trap(final_arr[i])):
                    final_arr[i]=new_solution
    #print(final_arr)
    return final_arr
    
#Algorithm 6
def lamc_aco(f,g,sigma):
    '''f (f==1 or f==3 or f==4 or f==5):
        f=80
        if(f==1):
            a = np.random.uniform(0, 30, size = f)
            b = np.empty(f)
            for i in range(0,f):
                x = five_uneven_peak_trap(a[i])
                b[i]=x
        elif(f==2):
            a = np.random.uniform(0,1,size=f)
            b = np.empty(f)
            for i in range(0,f):
                x = equal_maxima(a[i])
                b[i]=x
        '''
    if (f==1):
        f=80
        a = np.random.uniform(0, 30, size = f)
        b = np.empty(f)
        for i in range(0,f):
            x = five_uneven_peak_trap(a[i])
            b[i]=x
    f_max,f_min=b[0],b[0];
    for i in range(0,f):
        if(b[i]>f_max):
            f_max=b[i]
        elif(b[i]<f_min):
            f_min=b[i]
    y=randint(0,18)
    Clustersize=g[y]
    sorted_arr = np.empty(f)
    sorted_arr=clustering_for_crowding(a,Clustersize)
    
    new_soloutions_arr= np.empty(f)
    new_soloutions_arr=solution_contruction(Clustersize,sorted_arr,f_max,f_min)
    
    final_arr=np.empty(f)
    for i in range(0,f-1):
        if(five_uneven_peak_trap(sorted_arr[i])>five_uneven_peak_trap(new_soloutions_arr[i])):
            final_arr[i]=sorted_arr[i]
        elif(five_uneven_peak_trap(sorted_arr[i])<five_uneven_peak_trap(new_soloutions_arr[i])):
            final_arr[i]=new_soloutions_arr[i]
            
    c = np.empty(f)
    for i in range(0,f-1):
        y = float(five_uneven_peak_trap(final_arr[i]))
        c[i]=float(y)
    result_arr = np.empty(f)
    result_arr=adaptive_local_search(final_arr,len(final_arr),c,0.0001,2)
    print("The result population after applying Algorithm 6(LAMC-ACO) is:" )
    print(result_arr)
    
    result_fitness= np.empty(f)
    for i in range(0,f-1):
        z = float(five_uneven_peak_trap(result_arr[i]))
        result_fitness[i]=float(z)
    print("The result sorted fitness after applying Algorithm 6(LAMC-ACO) is:" )
    result_fitness[::-1].sort()
    print(result_fitness)

#Algorithm 7
def lams_aco(f,g,sigma):
    
    if (f==1):
        f=80
        a = np.random.uniform(0, 30, size = f)
        b = np.empty(f)
        for i in range(0,f-1):
            x = float(five_uneven_peak_trap(a[i]))
            b[i]=float(x)
    f_max,f_min=b[0],b[0];
    for i in range(0,f-1):
        if(b[i]>f_max):
            f_max=b[i]
        elif(b[i]<f_min):
            f_min=b[i]
   
    
    y=randint(0,18)
    Clustersize=g[y]
    
    sorted_arr = np.empty(f)
    sorted_arr=clustering_for_speciation(a,Clustersize)
    
    #print(sorted_arr)
    
    new_soloutions_arr= np.empty(f)
    new_soloutions_arr=solution_contruction(Clustersize,sorted_arr,f_max,f_min)
    
    final_arr=np.empty(f)
    for i in range(0,f-1):
        if(five_uneven_peak_trap(sorted_arr[i])>five_uneven_peak_trap(new_soloutions_arr[i])):
            final_arr[i]=sorted_arr[i]
        elif(five_uneven_peak_trap(sorted_arr[i])<five_uneven_peak_trap(new_soloutions_arr[i])):
            final_arr[i]=new_soloutions_arr[i]
    
    c = np.empty(f)
    for i in range(0,f-1):
        y = float(five_uneven_peak_trap(final_arr[i]))
        c[i]=float(y)
    result_arr = np.empty(f)
    result_arr=adaptive_local_search(final_arr,len(final_arr),c,0.0001,2)
    print("The result population after applying Algorithm 7(LAMS-ACO) is:" )
    print(result_arr)
    
    result_fitness= np.empty(f)
    for i in range(0,f-1):
        z = float(five_uneven_peak_trap(result_arr[i]))
        result_fitness[i]=float(z)
    print("The result sorted fitness after applying Algorithm 7(LAMS-ACO) is:" )
    result_fitness[::-1].sort()
    print(result_fitness)
    
#Implementing the Algorithms
f=float(input("which function do you want to apply the algorithm on from 1 to 12: "))
#name=f
lamc_aco(f,np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),0.0001)
lams_aco(f,np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),0.0001)
#End Implementation