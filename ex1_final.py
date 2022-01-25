# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 10:44:24 2022

HOMEWORK for Tensor Networks. Master in Quantum Science and Technology.
Universitat de Barcelona.

@author: Jofre Vallès Muns -> https://github.com/jofrevalles

"""
import numpy as np
import matplotlib.pyplot as plt


#TN functions -> The functions that compute the results

def norm_value(L,M):
    """
    Returns the value of the norm for a given L (length) and M matrix.
    """
    mps = np.zeros([L-1,2,2])
    for i in range(L-1):
        mps[i,:,:] = M

    mps_0 = np.array([1,1])
    mps_L = np.array([1,1])


    #We find the norm:
    contraction_0 = np.tensordot(mps_0,mps[0,:,:],axes=(0,1))
    contraction = contraction_0
    for i in range(1,L-1):
        contraction = np.tensordot(contraction,mps[i,:,:,],axes=(0,1))
    contraction_end = np.tensordot(contraction,mps_L,axes=(0,0))

    return contraction_end

def prob_1_zero(r,L,M,norm):
    """
    Returns the probability of getting a zero in r.
    """
    zero_vector = np.array([1,0])
    plus_vector = np.array([1,1])

    M2 = M**2

    #Probability of one 0 in distance r (r=0,1,...,L-1)
    vector_l = plus_vector
    vector_r = plus_vector
    if(r>1):
        for i in range(r-1):
            vector_l = np.einsum("i,ij",vector_l,M2)
            # print("looping1")
    if(r<(L-2)):
        for i in range(L-1,r+1,-1):
            vector_r = np.einsum("ij,j",M2,vector_r)
            # print("looping2")
    if(r<=1):
        if(r==0):
            main = np.einsum("ij,kj,i,k,j",M,M,zero_vector,zero_vector,vector_r)
        else:
            main = np.einsum("ij,jk,il,lk,j,l,k",M,M,M,M,zero_vector,zero_vector,vector_r)
            # main = np.einsum("ij,j,jk,k,il,l,lk",M,zero_vector,M,vector_r,M,zero_vector,M)
    elif(r>=(L-2)):
        if(r==(L-1)):
            main = np.einsum("ij,ik,i,k,j",M,M,vector_l,zero_vector,zero_vector)
        else:
            main = np.einsum("ij,jk,il,lk,i,j,l",M,M,M,M,vector_l,zero_vector,zero_vector)
    else:
        main = np.einsum("i,ij,j,jk,k,il,l,lk",vector_l,M,zero_vector,M,vector_r,M,zero_vector,M)

    prob = main/norm

    return prob

def prob_2_zeros(r1,r2,L,M,norm):
    """
    Returns the probability of getting a zero in r1 and a zero in r2.
    """
    zero_vector = np.array([1,0])
    plus_vector = np.array([1,1])
    M2 = M**2

    #initialize
    vector_l = plus_vector
    vector_r = plus_vector
    central_matrix = np.array([[1,0],[0,1]])

    #We supose that r1 =/= r2

    #Contract left part -> vector. Only existists if r1>0 and r2>0
    if(r1>0 and r2>0):
        for i in range(min(r1,r2)-1):
            vector_l = np.einsum("j,ij->i",vector_l,M2)
            # print("loop1")

    #Contract R1 part -> vector(if r1=0 or r1=L-1)/matrix
    if(r1>0 and r1<L-1):
        matrix_r1 = np.einsum("ij,jk,il,lk,j,l->ik",M,M,M,M,zero_vector,zero_vector)
    elif(r1==0):
        vector_r1 = np.einsum("ij,kj,i,k->j",M,M,zero_vector,zero_vector)
    elif(r1==L-1):
        vector_r1 = np.einsum("ij,ik,j,k->i",M,M,zero_vector,zero_vector)

    #Contract central part -> matriu. Only existists if |r1-r2|>1
    if(abs(r1-r2)>1):
        for i in range(abs(r1-r2)-2):
            central_matrix = np.einsum("ij,jk->ik",central_matrix,M2)
            # print("loop2")

    #Contract R2 part -> vector(if r2=0 or r2=L-1)/matrix
    if(r2>0 and r2<L-1):
        matrix_r2 = np.einsum("ij,jk,il,lk,j,l->ik",M,M,M,M,zero_vector,zero_vector)
    elif(r2==0):
        vector_r2 = np.einsum("ij,kj,i,k->j",M,M,zero_vector,zero_vector)
    elif(r2==L-1):
        vector_r2 = np.einsum("ij,ik,j,k->i",M,M,zero_vector,zero_vector)

    #Contract right part -> vector. Only exists if r1<L-1 and r2<L-1
    if(r1<L-1 and r2<L-1):
        for i in range(max(r1,r2)+1,L-1):
            vector_r = np.einsum("ij,j->i",M2,vector_r)
            # print("loop3")


    #Contract all together:
    if(r1>0 and r2>0): #--> There is a left vector
        if(r1<L-1 and r2<L-1): #--> There is a right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,jk,kl,l",vector_l,matrix_r1,central_matrix,matrix_r2,vector_r)
                else:
                    contraction = np.einsum("i,ij,jk,kl,l",vector_l,matrix_r2,central_matrix,matrix_r1,vector_r)
            else: #--> There is no central matrix
                contraction = np.einsum("i,ij,jk,kl,im,mn,nl,j,k,m,n,l",vector_l,M,M,M,M,M,M,zero_vector,zero_vector,zero_vector,zero_vector,vector_r)
        else:#--> There is no right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,jk,k",vector_l,matrix_r1,central_matrix,vector_r2)
                else:
                    contraction = np.einsum("i,ij,jk,k",vector_l,matrix_r2,central_matrix,vector_r1)
            else: #--> There is no central_matrix
                contraction  = np.einsum("i,ij,jk,im,mn,j,k,m,n",vector_l,M,M,M,M,zero_vector,zero_vector,zero_vector,zero_vector)
    else: #--> There is no left vector
        if(r1<L-1 and r2<L-1): #--> There is a right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,jk,k",vector_r1,central_matrix,matrix_r2,vector_r)
                else:
                    contraction = np.einsum("i,ij,jk,k",vector_r2,central_matrix,matrix_r1,vector_r)
            else: #--> There is no central matrix
                contraction = np.einsum("ij,jk,lm,mk,i,j,l,m,k",M,M,M,M,zero_vector,zero_vector,zero_vector,zero_vector,vector_r)
        else: #--> There is no right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,j",vector_r1,central_matrix,vector_r2)
                else:
                    contraction = np.einsum("i,ij,j",vector_r2,central_matrix,vector_r1)
            else: #--> There is no central_matrix
                contraction = np.einsum("ij,kl,i,j,k,l",M,M,zero_vector,zero_vector,zero_vector,zero_vector)

    prob = contraction/norm

    return prob

def prob_2(r1,r2,i1,i2,L,M,norm):
    """
    Returns the probability of getting a i1 value in r1 and a i2 value in r2.
    """
    zero_vector = np.array([1,0])
    one_vector = np.array([0,1])

    if(i1 == 0):
        v1_vector = zero_vector
    elif(i1 == 1):
        v1_vector = one_vector
    if(i2 == 0):
        v2_vector = zero_vector
    elif(i1 == 1):
        v2_vector = one_vector

    plus_vector = np.array([1,1])
    M2 = M**2

    #initialize
    vector_l = plus_vector
    vector_r = plus_vector
    central_matrix = np.array([[1,0],[0,1]])

    #We suppose that r1 =/= r2

    #Contract left part -> vector. Only existists if r1>0 and r2>0
    if(r1>0 and r2>0):
        for i in range(min(r1,r2)-1):
            vector_l = np.einsum("j,ij->i",vector_l,M2)
            # print("loop1")

    #Contract R1 part -> vector(if r1=0 or r1=L-1)/matrix
    if(r1>0 and r1<L-1):
        matrix_r1 = np.einsum("ij,jk,il,lk,j,l->ik",M,M,M,M,v1_vector,v1_vector)
    elif(r1==0):
        vector_r1 = np.einsum("ij,kj,i,k->j",M,M,v1_vector,v1_vector)
    elif(r1==L-1):
        vector_r1 = np.einsum("ij,ik,j,k->i",M,M,v1_vector,v1_vector)

    #Contract central part -> matrix. Only exists if |r1-r2|>1
    if(abs(r1-r2)>1):
        for i in range(abs(r1-r2)-2):
            central_matrix = np.einsum("ij,jk->ik",central_matrix,M2)
            # print("loop2")

    #Contract R2 part -> vector(if r2=0 or r2=L-1)/matrix
    if(r2>0 and r2<L-1):
        matrix_r2 = np.einsum("ij,jk,il,lk,j,l->ik",M,M,M,M,v2_vector,v2_vector)
    elif(r2==0):
        vector_r2 = np.einsum("ij,kj,i,k->j",M,M,v2_vector,v2_vector)
    elif(r2==L-1):
        vector_r2 = np.einsum("ij,ik,j,k->i",M,M,v2_vector,v2_vector)

    #Contract right part -> vector. Only exists if r1<L-1 and r2<L-1
    if(r1<L-1 and r2<L-1):
        for i in range(max(r1,r2)+1,L-1):
            vector_r = np.einsum("ij,j->i",M2,vector_r)
            # print("loop3")


    #Contract all together:
    if(r1>0 and r2>0): #--> There is a left vector
        if(r1<L-1 and r2<L-1): #--> There is a right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,jk,kl,l",vector_l,matrix_r1,central_matrix,matrix_r2,vector_r)
                else:
                    contraction = np.einsum("i,ij,jk,kl,l",vector_l,matrix_r2,central_matrix,matrix_r1,vector_r)
            else: #--> There is no central matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,jk,kl,im,mn,nl,j,k,m,n,l",vector_l,M,M,M,M,M,M,v1_vector,v2_vector,v1_vector,v2_vector,vector_r)
                else:
                    contraction = np.einsum("i,ij,jk,kl,im,mn,nl,j,k,m,n,l",vector_l,M,M,M,M,M,M,v2_vector,v1_vector,v2_vector,v1_vector,vector_r)
        else:#--> There is no right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,jk,k",vector_l,matrix_r1,central_matrix,vector_r2)
                else:
                    contraction = np.einsum("i,ij,jk,k",vector_l,matrix_r2,central_matrix,vector_r1)
            else: #--> There is no central_matrix
                if(r2>r1):
                    contraction  = np.einsum("i,ij,jk,im,mn,j,k,m,n",vector_l,M,M,M,M,v1_vector,v2_vector,v1_vector,v2_vector)
                else:
                    contraction  = np.einsum("i,ij,jk,im,mn,j,k,m,n",vector_l,M,M,M,M,v2_vector,v1_vector,v2_vector,v1_vector)
    else: #--> There is no left vector
        if(r1<L-1 and r2<L-1): #--> There is a right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,jk,k",vector_r1,central_matrix,matrix_r2,vector_r)
                else:
                    contraction = np.einsum("i,ij,jk,k",vector_r2,central_matrix,matrix_r1,vector_r)
            else: #--> There is no central matrix
                if(r2>r1):
                    contraction = np.einsum("ij,jk,lm,mk,i,j,l,m,k",M,M,M,M,v1_vector,v2_vector,v1_vector,v2_vector,vector_r)
                else:
                    contraction = np.einsum("ij,jk,lm,mk,i,j,l,m,k",M,M,M,M,v2_vector,v1_vector,v2_vector,v1_vector,vector_r)
        else: #--> There is no right vector
            if(abs(r1-r2)>1): #--> There is central_matrix
                if(r2>r1):
                    contraction = np.einsum("i,ij,j",vector_r1,central_matrix,vector_r2)
                else:
                    contraction = np.einsum("i,ij,j",vector_r2,central_matrix,vector_r1)
            else: #--> There is no central_matrix
                if(r2>r1):
                    contraction = np.einsum("ij,kl,i,j,k,l",M,M,v1_vector,v2_vector,v1_vector,v2_vector)
                else:
                    contraction = np.einsum("ij,kl,i,j,k,l",M,M,v2_vector,v1_vector,v2_vector,v1_vector)

    prob = contraction/norm

    return prob


# Plots. We will use L=20 for this two first sections of the homework.

#%% 1st section -> Probability of finding two bits in 0 at distance r <= L.

L = 20
M = np.array([[1,1],[1,0]])
norm = norm_value(L,M)
prob_r1_r2 = np.zeros([L,L])
r1 = np.zeros([L,L],dtype = np.int)
r2 = np.zeros([L,L],dtype = np.int)

#Prob. of two bits in zero:
i1 = 0
i2 = 0

for i in range(L):
    for j in range(L):
        r1[i,j] = int(i)
        r2[i,j] = int(j)

for j in range(L):
    for i in range(L):
        if(r2[j,i] != r1[j,i]):
            prob_r1_r2[j,i] = prob_2(r1[j,i], r2[j,i], i1, i2, L, M, norm)
        else:
            prob_r1_r2[j,i] = None

fig, axs = plt.subplots(2, 2)

#Starting position for r1 for each suplot:
r1_1 = 0
r1_2 = 1
r1_3 = 2
r1_4 = 3

fig.suptitle(r"$P(r_{2}-r_{1})$ of 2 bits in Zero", fontsize=14)

axs[0,0].plot(r2[r1_1,:]-r1[r1_1,:],prob_r1_r2[r1_1,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_1,0])+"$")
axs[0,1].plot(r2[r1_2,:]-r1[r1_2,:],prob_r1_r2[r1_2,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_2,0])+"$")
axs[1,0].plot(r2[r1_3,:]-r1[r1_3,:],prob_r1_r2[r1_3,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_3,0])+"$")
axs[1,1].plot(r2[r1_4,:]-r1[r1_4,:],prob_r1_r2[r1_4,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_4,0])+"$")
axs[1,0].set_xlabel(r"$r_{2}-r_{1}$",fontsize=14)
axs[1,1].set_xlabel(r"$r_{2}-r_{1}$",fontsize=14)
axs[0,0].set_ylabel(r"$p(r_{2}-r_{1})$",fontsize=14)
axs[1,0].set_ylabel(r"$p(r_{2}-r_{1})$",fontsize=14)

# for ax in axs.flat:
    # ax.label_outer()

axs[0,0].legend(loc="lower center")
axs[0,1].legend(loc="lower center")
axs[1,0].legend(loc="lower center")
axs[1,1].legend(loc="lower center")

plt.subplots_adjust(top=0.923,
bottom=0.098,
left=0.108,
right=0.987,
hspace=0.216,
wspace=0.23)

# plt.savefig('Prob_2bits_in_zero_4_plots.eps')

#%% Bonus: Colormap of the first section

L = 20
M = np.array([[1,1],[1,0]])
norm = norm_value(L,M)


r1 = np.arange(0,L+1,1,dtype=np.int)
r2 = np.arange(0,L+1,1,dtype=np.int)
i1 = 0
i2 = 0

prob_r1_r2 = np.zeros([L,L])

maximum_p = 0
minimum_p = 100

for i in range(L):
    for j in range(L):
        if(i==j):
            prob_r1_r2[i,j] = None
        else:
            prob_r1_r2[i,j] = prob_2(r1[i], r2[j], i1, i2, L, M, norm)
            if(prob_r1_r2[i,j] > maximum_p):
                maximum_p = prob_r1_r2[i,j]
            if(prob_r1_r2[i,j] < minimum_p):
                minimum_p = prob_r1_r2[i,j]

fig, ax = plt.subplots()
fig.suptitle("Probabilty of two bits in Zero", fontsize=16)
plt.pcolormesh(r1,r2,prob_r1_r2,cmap="viridis")
ax.set_xlabel(r"$r_{1}$",fontsize=15)
ax.set_ylabel(r"$r_{2}$",fontsize=15)
cbar = plt.colorbar()
plt.clim(minimum_p,maximum_p)
cbar.set_label(r"$p(r_{1},r_{2})$",fontsize=15)

plt.subplots_adjust(top=0.918,
bottom=0.108,
left=0.108,
right=0.965,
hspace=0.216,
wspace=0.23)

# plt.savefig('Colormap_2bits_in_zeros.eps')


#%% 2nd section -> Probability of finding two bits in 1 at distance r <= L.

L = 20
M = np.array([[1,1],[1,0]])
norm = norm_value(L,M)
prob_r1_r2 = np.zeros([L,L])
r1 = np.zeros([L,L],dtype = np.int)
r2 = np.zeros([L,L],dtype = np.int)

#Prob. of two bits in one:
i1 = 1
i2 = 1

for i in range(L):
    for j in range(L):
        r1[i,j] = int(i)
        r2[i,j] = int(j)

for j in range(L):
    for i in range(L):
        if(r2[j,i] != r1[j,i]):
            prob_r1_r2[j,i] = prob_2(r1[j,i], r2[j,i], i1, i2, L, M, norm)
        else:
            prob_r1_r2[j,i] = None

fig, axs = plt.subplots(2, 2)

#Starting position for r1 for each suplot:
r1_1 = 0
r1_2 = 1
r1_3 = 2
r1_4 = 3

fig.suptitle(r"$P(r_{2}-r_{1})$ of 2 bits in One", fontsize=14)

axs[0,0].plot(r2[r1_1,:]-r1[r1_1,:],prob_r1_r2[r1_1,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_1,0])+"$", color="C1")
axs[0,1].plot(r2[r1_2,:]-r1[r1_2,:],prob_r1_r2[r1_2,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_2,0])+"$", color="C1")
axs[1,0].plot(r2[r1_3,:]-r1[r1_3,:],prob_r1_r2[r1_3,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_3,0])+"$", color="C1")
axs[1,1].plot(r2[r1_4,:]-r1[r1_4,:],prob_r1_r2[r1_4,:],"o",linestyle="-",label=r"$r_{1}="+str(r1[r1_4,0])+"$", color="C1")
axs[1,0].set_xlabel(r"$r_{2}-r_{1}$",fontsize=14)
axs[1,1].set_xlabel(r"$r_{2}-r_{1}$",fontsize=14)
axs[0,0].set_ylabel(r"$p(r_{2}-r_{1})$",fontsize=14)
axs[1,0].set_ylabel(r"$p(r_{2}-r_{1})$",fontsize=14)

# for ax in axs.flat:
    # ax.label_outer()

axs[0,0].legend(loc="lower center")
axs[0,1].legend(loc="lower center")
axs[1,0].legend(loc="lower center")
axs[1,1].legend(loc="lower center")

plt.subplots_adjust(top=0.923,
bottom=0.098,
left=0.108,
right=0.987,
hspace=0.216,
wspace=0.23)

# plt.savefig('Prob_2bits_in_ones_4_plots.eps')

#%% Bonus: Colormap of the second section

L = 20
M = np.array([[1,1],[1,0]])
norm = norm_value(L,M)


r1 = np.arange(0,L+1,1,dtype=np.int)
r2 = np.arange(0,L+1,1,dtype=np.int)
i1 = 1
i2 = 1

prob_r1_r2 = np.zeros([L,L])

maximum_p = 0
minimum_p = 100

for i in range(L):
    for j in range(L):
        if(i==j):
            prob_r1_r2[i,j] = None
        else:
            prob_r1_r2[i,j] = prob_2(r1[i], r2[j], i1, i2, L, M, norm)
            if(prob_r1_r2[i,j] > maximum_p):
                maximum_p = prob_r1_r2[i,j]
            if(prob_r1_r2[i,j] < minimum_p):
                minimum_p = prob_r1_r2[i,j]

fig, ax = plt.subplots()
fig.suptitle("Probabilty of two bits in One", fontsize=16)
plt.pcolormesh(r1,r2,prob_r1_r2,cmap="viridis")
ax.set_xlabel(r"$r_{1}$",fontsize=15)
ax.set_ylabel(r"$r_{2}$",fontsize=15)
cbar = plt.colorbar()
plt.clim(minimum_p,maximum_p)
cbar.set_label(r"$p(r_{1},r_{2})$",fontsize=15)

plt.subplots_adjust(top=0.918,
bottom=0.108,
left=0.108,
right=0.965,
hspace=0.216,
wspace=0.23)

# plt.savefig('Colormap_2bits_in_ones.eps')

#%% 3rd section -> We will work in L=8
L = 8

#We have found that this M_prime will work well for our need
M_prime = np.array([[np.exp(-20),np.exp(-20)],[1,1]])

#We now contract all the M_prime matrices to obtain the tensor of components
C = np.einsum("ij,jk,kl,lm,mn,no,op->ijklmnop",M_prime,M_prime,M_prime,M_prime,M_prime,M_prime,M_prime)
total_sum = np.sum(C**2)

#This vector will save the probability of having 2 ones in r1=0, r2=L-1 given having 0,1,2,3,4,5,6 zeros in between
vector_p = np.zeros([L-1])

vector_p[0] = C[1,1,1,1,1,1,1,1]/total_sum
vector_p[1] = (C[1,0,1,1,1,1,1,1]+C[1,1,0,1,1,1,1,1]+C[1,1,1,0,1,1,1,1]+C[1,1,1,1,0,1,1,1]+C[1,1,1,1,1,0,1,1]+C[1,1,1,1,1,1,0,1])/total_sum
vector_p[2] = (C[1,0,0,1,1,1,1,1]+C[1,0,1,0,1,1,1,1]+C[1,0,1,1,0,1,1,1]+C[1,0,1,1,1,0,1,1]+C[1,0,1,1,1,1,0,1]+C[1,1,0,0,1,1,1,1]+C[1,1,0,1,0,1,1,1]+C[1,1,0,1,1,0,1,1]+C[1,1,0,1,1,1,0,1]+C[1,1,1,0,0,1,1,1]+C[1,1,1,0,1,0,1,1]+C[1,1,1,0,1,1,0,1]+C[1,1,1,1,0,0,1,1]+C[1,1,1,1,0,1,0,1]+C[1,1,1,1,1,0,0,1])/total_sum
vector_p[3] = (C[1,0,0,0,1,1,1,1]+C[1,0,0,1,0,1,1,1]+C[1,0,0,1,1,0,1,1]+C[1,0,0,1,1,1,0,1]+C[1,0,1,0,0,1,1,1]+C[1,0,1,0,1,0,1,1]+C[1,0,1,0,1,1,0,1]+C[1,0,1,1,0,0,1,1]+C[1,0,1,1,0,1,0,1]+C[1,0,1,1,1,0,0,1]+C[1,1,0,0,0,1,1,1]+C[1,1,0,0,1,0,1,1]+C[1,1,0,0,1,1,0,1]+C[1,1,0,1,0,0,1,1]+C[1,1,0,1,0,1,0,1]+C[1,1,0,1,0,0,1,1]+C[1,1,0,1,0,1,0,1]+C[1,1,0,1,1,0,0,1]+C[1,1,1,0,0,0,1,1]+C[1,1,1,0,0,1,0,1]+C[1,1,1,0,1,0,0,1]+C[1,1,1,1,0,0,0,1])/total_sum
vector_p[4] = (C[1,0,0,0,0,1,1,1]+C[1,0,0,0,1,0,1,1]+C[1,0,0,0,1,1,0,1]+C[1,0,0,1,0,0,1,1]+C[1,0,0,1,0,1,0,1]+C[1,0,0,1,1,0,0,1]+C[1,0,1,0,0,0,1,1]+C[1,0,1,0,0,1,0,1]+C[1,0,1,0,1,0,0,1]+C[1,0,1,1,0,0,0,1]+C[1,1,0,0,0,0,1,1]+C[1,1,0,0,0,1,0,1]+C[1,1,0,0,1,0,0,1]+C[1,1,0,1,0,0,0,1]+C[1,1,1,0,0,0,0,1])/total_sum
vector_p[5] = (C[1,0,0,0,0,0,1,1]+C[1,0,0,0,0,1,0,1]+C[1,0,0,0,1,0,0,1]+C[1,0,0,1,0,0,0,1]+C[1,0,1,0,0,0,0,1]+C[1,1,0,0,0,0,0,1])/total_sum
vector_p[6] = C[1,0,0,0,0,0,0,1]/total_sum

fig, ax = plt.subplots()
fig.suptitle("Probability of 2 bits in One. Linear scale", fontsize=16)
ax.plot(np.arange(0,len(vector_p),1),vector_p,linestyle="-",label=r"$r_{1}=0, r_{2}=7$")
ax.set_xlabel(r"$N_{zeros}$",fontsize=15)
ax.set_ylabel(r"$P(r_{1},r_{2},N_{zeros})$",fontsize=15)
ax.legend(loc="best")
plt.subplots_adjust(top=0.913,
bottom=0.113,
left=0.148,
right=0.957,
hspace=0.216,
wspace=0.23)
# plt.savefig('Prob_2bits_in_one_given_zero_in_between__linear.eps')

fig, ax = plt.subplots()
fig.suptitle("Probability of 2 bits in One. Logarithmic scale", fontsize=16)
ax.plot(np.arange(0,len(vector_p),1),vector_p,linestyle="-",label=r"$r_{1}=0, r_{2}=7$")
ax.set_xlabel(r"$N_{zeros}$",fontsize=15)
ax.set_ylabel(r"$P(r_{1},r_{2},N_{zeros})$",fontsize=15)
ax.set_yscale('log')
ax.legend(loc="best")
plt.subplots_adjust(top=0.913,
bottom=0.113,
left=0.148,
right=0.957,
hspace=0.216,
wspace=0.23)
# plt.savefig('Prob_2bits_in_one_given_zero_in_between__log.eps')