from pyomo.environ import *
from pyomo.common.timing import TicTocTimer
from time import process_time
import numpy as np
import matplotlib.pyplot as plt
model = AbstractModel()
data = 'Data200_profileA.dat'
t1_start = process_time()

# SETS
model.c = Set() # Leukapheresis sites
model.h = Set() # Hospitals
model.j = Set() # Transport mode
model.m = Set() # Manufacturing sites
model.p = Set() # Patients
model.t = RangeSet(130) # Time
model.tt = Set(initialize=model.t) # Alias of set t

# Indexed PARAMETERS
model.CIM = Param(model.m) # Capital investment for manufacturing facility
model.FCAP = Param(model.m) # Total capacity of manufacturing site
model.TT1 = Param(model.j) # Transport time LS to MS using transport mode j
model.TT3 = Param(model.j) # Transport time MS to hospital using transport mode j
model.U1 = Param(model.c, model.m, model.j) # Unit transport cost LS to MS using transport mode j
model.U3 = Param(model.m, model.h, model.j) # Unit transport cost MS to hospital using transport mode j
model.INC = Param(model.p, model.c, model.t, initialize=0) # Demand therapy p arriving for leukapheresis at LS site c at time t
model.CVM = Param(model.m, default={'m1':20920, 'm2':156900, 'm3':52300, 'm4':20920, 'm5':156900, 'm6':52300}) #Fixed variable costs


# Scalar PARAMETERS
model.FMAX = Param() # Maximum flow
model.FMIN = Param() # Minimum flow
model.TAD = Param(within=NonNegativeReals) # Duration of administration
model.TLS = Param(within=NonNegativeReals) # Duration of leukapheresis
model.TMFE = Param(default=7) # Duration of manufacturing excluding QC
model.TQC = Param(default=7) # Duration of QC
model.C_material = Param(default=10476) # Materials cost per therapy
model.CQC= Param(default=9312) # QC cost per therapy
model.ND = Param(default = 18) # Maximum turnaround (vein-to-vein) time

# Binary VARIABLES
model.E1 = Var(model.m, within=Binary) # 1 if manufacturing facility m is established
model.X1 = Var(model.c, model.m) # 1 if a match between LS site c and MS site m is established - UNIMODULARITY no need to define as binary
model.X2 = Var(model.m, model.h) # 1 if a match between MS site m and a hospital h is established - UNIMODULARITY no need to define as binary
model.Y1 = Var(model.p, model.c, model.m, model.j, model.t, within=Binary) # 1 if a sample p is transferred from a LS site c to a MS site m via mode j at time t
model.Y2 = Var(model.p, model.m, model.h, model.j, model.t, within=Binary) # 1 if a sample p is transferred from a MS site m to a hospital h via mode j at time t

# Integer variables
model.INH = Var(model.p, model.h, model.t, within=NonNegativeIntegers) # Therapy p arriving at hospital h at time t

# Positive VARIABLES
model.CTM = Var(model.p, within=NonNegativeReals)
model.FTD = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
model.TTC = Var(model.p, within=NonNegativeReals)
model.LSA = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
model.LSR = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
model.MSO = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
model.OUTC = Var(model.p, model.c, model.t, within=NonNegativeReals)
model.OUTM = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.INM = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.DURV = Var(model.p, model.m, model.t, within=NonNegativeReals) # 1 only for the time period t in which a therapy p is manufactured in facility m
model.RATIO = Var(model.m, model.t, within=NonNegativeReals) # the percentage of utilisation of MS site m at time t


# VARIABLES
model.CAP = Var(model.m, model.t) # Capacity of MS m at time t
model.TRT = Var(model.p) # Total return time of therapy
model.ATRT = Var() # Average return time
model.STT = Var(model.p) # Starting time of treatment for patient p
model.CTT = Var(model.p) # Completion time of treatment for patient p


# OBJECTIVE FUNCTION
def obj_rule(model):
    return sum( model.CTM[p]for p in model.p )+ sum( model.TTC[p] for p in model.p ) + (model.C_material + model.CQC)* len(model.p)
model.obj = Objective( rule=obj_rule )


# CONSTRAINTS
# Manufacturing cost
def C1_rule(model,p):
    return model.CTM[p] == sum((model.E1[m]*(model.CIM[m]+model.CVM[m]))*len(model.t)/len(model.p) for m in model.m)
model.C1 = Constraint(model.p, rule=C1_rule)

# Transportation cost
def C2_rule(model,p):
    return model.TTC[p] == sum(model.Y1[p,c,m,j,t]*model.U1[c,m,j] for c in model.c for m in model.m for j in model.j for t in model.t) + sum(model.Y2[p,m,h,j,t]*model.U3[m,h,j] for m in model.m for h in model.h for j in model.j for t in model.t)
model.C2 = Constraint(model.p, rule=C2_rule)

#The percentage of utilisation of facility m at time t
def RATIOEQ_rule(model,m,t):
    return model.RATIO[m,t] == sum(model.DURV[p,m,t]/model.FCAP[m] for p in model.p)
model.RATIOEQ = Constraint(model.m, model.t, rule=RATIOEQ_rule)

def MSBnew_rule(model,p,m,t):
     return model.DURV[p,m,t] == sum(model.INM[p,m,tt-1]-model.OUTM[p,m,tt] for tt in model.tt if tt<=t and tt>1) + model.OUTM[p,m,t] 
model.MSBnew = Constraint(model.p, model.m, model.t, rule=MSBnew_rule)


# Material flow balances
#MSB1
def MSB1_rule(model,p,c,t,tt):
    if tt == t + model.TLS:
        return model.INC[p,c,t] == model.OUTC[p,c,tt]
    else:
        return Constraint.Skip
model.MSB1 = Constraint(model.p, model.c, model.t, model.tt, rule=MSB1_rule)

#MSB3
def MSB3_rule(model,p,c,m,j,t,tt):
    if tt == t + model.TT1[j]:
        return model.LSR[p,c,m,j,t] == model.LSA[p,c,m,j,tt]
    else:
        return Constraint.Skip
model.MSB3 = Constraint(model.p, model.c, model.m, model.j, model.t, model.tt, rule=MSB3_rule)

#MSB7
def MSB7_rule(model,p,c,t):
    return model.OUTC[p,c,t] == sum(model.LSR[p,c,m,j,t] for m in model.m for j in model.j)
model.MSB7 = Constraint(model.p, model.c, model.t, rule=MSB7_rule)

#MSB5
def MSB5_rule(model,p,m,t):
    return model.INM[p,m,t] == sum(model.LSA[p,c,m,j,t] for c in model.c for j in model.j)
model.MSB5 = Constraint(model.p, model.m, model.t, rule=MSB5_rule)

#MSB2
def MSB2_rule(model,p,m,t,tt):
    if tt == t + model.TMFE:
        return model.INM[p,m,t] == model.OUTM[p,m,tt]
    else:
        return Constraint.Skip
model.MSB2 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB2_rule)

#MSB8
def MSB8_rule(model,p,m,t,tt):
    if tt == t + model.TQC:
        return model.OUTM[p,m,t] == sum(model.MSO[p,m,h,j,tt] for h in model.h for j in model.j)
    else:
        return Constraint.Skip
model.MSB8 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB8_rule)


#MSB4
def MSB4_rule(model,p,m,h,j,t,tt):
    if tt == t + model.TT3[j]:
        return model.MSO[p,m,h,j,t] == model.FTD[p,m,h,j,tt]
    else:
        return Constraint.Skip
model.MSB4 = Constraint(model.p, model.m, model.h, model.j, model.t, model.tt, rule=MSB4_rule)

#MSB6
def MSB6_rule(model,p,h,t):
    return model.INH[p,h,t] == sum(model.FTD[p,m,h,j,t] for m in model.m for j in model.j)
model.MSB6 = Constraint(model.p, model.h, model.t, rule=MSB6_rule)


# Capacity constraints
def CAP1_rule(model,m,t):
    return model.CAP[m,t] == model.FCAP[m]-sum(model.INM[p,m,tt] for p in model.p for tt in model.tt if tt<t and tt>=t-model.TMFE)
model.CAP1 = Constraint(model.m, model.t, rule=CAP1_rule)


def CAPCON1_rule(model,m,t):
    return sum(model.INM[p,m,t] for p in model.p)-sum(model.OUTM[p,m,t] for p in model.p) <= model.CAP[m,t]
model.CAPCON1 = Constraint(model.m, model.t, rule=CAPCON1_rule)


# NETWORK STRUCTURE CONSTRAINTS
# Constraint to allow the establishment of up to 2 manufacturing facilities (centralized network). Relax this for the investigation of decentralization
#CON1
def CON1_rule(model):
    return sum(model.E1[m] for m in model.m) <= 2
model.CON1 = Constraint(rule=CON1_rule)

# Constraints to ensure that no matches are established with non-existent facilities
#CON2
def CON2_rule(model,c,m):
    return model.X1[c,m] <= model.E1[m]
model.CON2 = Constraint(model.c, model.m, rule=CON2_rule)


#CON3
def CON3_rule(model,m,h):
    return model.X2[m,h] <= model.E1[m]
model.CON3 = Constraint(model.m, model.h, rule=CON3_rule)

# Logical constraints for transportation flows
#CON4
def CON4_rule(model,p,c,m,j,t):
    return model.Y1[p,c,m,j,t] <= model.X1[c,m]
model.CON4 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON4_rule)

#CON5
def CON5_rule(model,p,m,h,j,t):
    return model.Y2[p,m,h,j,t] <= model.X2[m,h]
model.CON5 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON5_rule)

# Ensure that only one transport mode is selected for every therapy p at every journey can be selected
#CON6
def CON6_rule(model,p):
    return sum(model.Y1[p,c,m,j,t] for c in model.c for m in model.m for j in model.j for t in model.t) == 1
model.CON6 = Constraint(model.p, rule=CON6_rule)


#CON7
def CON7_rule(model,p):
    return sum(model.Y2[p,m,h,j,t] for m in model.m for h in model.h for j in model.j for t in model.t) == 1
model.CON7 = Constraint(model.p, rule=CON7_rule)


# Demand satisfaction
#DEM
def DEM_rule(model):
    return sum(model.INH[p,h,t] for p in model.p for h in model.h for t in model.t) <= len(model.p)
model.DEM = Constraint(rule=DEM_rule)


# Flow constraints
# These ensure that a minimum and maximum flow of material exists for a transportation link to be established
#CON8
def CON8_rule(model,p,c,m,j,t):
    return model.LSR[p,c,m,j,t] >= model.Y1[p,c,m,j,t]*model.FMIN
model.CON8 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON8_rule)


#CON9
def CON9_rule(model,p,c,m,j,t):
    return model.LSR[p,c,m,j,t] <= model.Y1[p,c,m,j,t]*model.FMAX
model.CON9 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON9_rule)


#CON10
def CON10_rule(model,p,m,h,j,t):
    return model.MSO[p,m,h,j,t] >= model.Y2[p,m,h,j,t]*model.FMIN
model.CON10 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON10_rule)


#CON11
def CON11_rule(model,p,m,h,j,t):
    return model.MSO[p,m,h,j,t] <= model.Y2[p,m,h,j,t]*model.FMAX
model.CON11 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON11_rule)

# These constraints make sure that a match is only made between a leukapheresis site c and its corresponding co-located hospital h
#CON12
def CON12_rule(model,p):
    return sum(model.Y2[p,m,'h1',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c1',t] for t in model.t)
model.CON12 = Constraint(model.p, rule=CON12_rule)


#CON13
def CON13_rule(model,p):
    return sum(model.Y2[p,m,'h2',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c2',t] for t in model.t)
model.CON13 = Constraint(model.p, rule=CON13_rule)


#CON14
def CON14_rule(model,p):
    return sum(model.Y2[p,m,'h3',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c3',t] for t in model.t)
model.CON14 = Constraint(model.p, rule=CON14_rule)


#CON15
def CON15_rule(model,p):
    return sum(model.Y2[p,m,'h4',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c4',t] for t in model.t)
model.CON15 = Constraint(model.p, rule=CON15_rule)


# Time constraints
#Maximum turnaround time
def TCON_rule(model,p):
    return model.TRT[p] <= model.ND
model.TCON = Constraint(model.p, rule=TCON_rule)


#START time of leukapheresis
def START_rule(model,p):
    return model.STT[p] == sum(model.INC[p,c,t]*t for c in model.c for t in model.t)
model.START = Constraint(model.p, rule=START_rule)


#END time of therapy administration
def END_rule(model,p):
    return model.CTT[p] == sum(model.INH[p,h,t]*t for h in model.h for t in model.t)
model.END = Constraint(model.p, rule=END_rule)

# Makes sure that the time point a patient checks into a leukapheresis site chronologically precedes the time point the corresponding therapy p is delivered to the hospital 
def TSEQ_rule(model,p):
    return model.STT[p] <= model.CTT[p]
model.TSEQ = Constraint(model.p, rule=TSEQ_rule)


# Total turnaround time
def TIME_rule(model,p):
    return model.TRT[p] == model.CTT[p] - model.STT[p]
model.TIME = Constraint(model.p, rule=TIME_rule)


#Average turnaround time
def ATIME_rule(model):
    return model.ATRT == sum(model.TRT[p] for p in model.p)/len(model.p)
model.ATIME = Constraint(rule=ATIME_rule)



timer = TicTocTimer()
timer.tic('start')
#report_timing()
print('-----------------------------------------------Building model-----------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------')
instance = model.create_instance(data)
timer.toc('Built model')




print('-----------------------------------------------Solving model------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------')
opt = SolverFactory('gurobi')
myoptions = dict()
myoptions = {'OutputFlag': 1}  #Gurobi log

#myoptions['timelimit'] = 2
results = opt.solve(instance, options=myoptions, tee=True) # solves and updates instance
timer.toc('Time to solve')
t1_stop = process_time()


print('------------------------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------RESULTS---------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------RATIO(m,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for t in instance.t:
    for m in instance.m:
        if value(instance.RATIO[m,t])*100 > 1e-3:
            print('The percentage of utilisation of MS site', m, 'at time','t{}'.format(t), 'is', value(instance.RATIO[m,t])*100,'%')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------OUTC(p,c,t)------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for c in instance.c:
        for t in instance.t:
            if value(instance.OUTC[p,c,t]) == 1:
                print('Therapy',p,'leaving leukapheresis site',c,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------LSR(p,c,m,j,t)---------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for c in instance.c:
        for m in instance.m:
            for j in instance.j:
                for t in instance.t:
                    if value(instance.LSR[p,c,m,j,t]) == 1:
                        print('Therapy',p,'leaving LS',c, 'arriving at MS',m,'with transport mode',j,'at time','t{}'.format(t))                
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------INM(p,m,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for t in instance.t:
            if value(instance.INM[p,m,t]) == 1:
                print('Therapy',p,'entering manufacturing site',m,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')                
print('------------------------------------------------DURV(p,m,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for t in instance.t:
            if t<130:
                if (value(instance.DURV[p,m,t+1])-value(instance.DURV[p,m,t]))**2 == 1:
                    print('Therapy',p,'is manufactured at site',m,'for the time period','t{}'.format(t),'-','t{}'.format(t+ instance.TMFE))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------OUTM(p,m,t)------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for t in instance.t:
            if value(instance.OUTM[p,m,t]) == 1:
                print('Therapy',p,'leaving manufacturing site', m,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')                
print('------------------------------------------------LSA(p,c,m,j,t)---------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for c in instance.c:
        for m in instance.m:
            for j in instance.j:
                for t in instance.t:
                    if value(instance.LSA[p,c,m,j,t]) == 1:
                        print('Therapy',p,'that left LS', c,'arriving at MS',m,'with transport mode', j,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------MSO(p,m,h,j,t)----------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for h in instance.h:
            for j in instance.j:
                for t in instance.t:
                    if value(instance.MSO[p,m,h,j,t]) == 1:
                        print('Therapy',p,'leaving MS',m, 'arriving at hospital',h, 'with transport mode',j,'at time','t{}'.format(t))
                        
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')                 
print('------------------------------------------------FTD(p,m,h,j,t)----------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for h in instance.h:
            for j in instance.j:
                for t in instance.t:
                    if value(instance.FTD[p,m,h,j,t]) == 1:
                        print('Final therapy',p,'that left MS',m,'arriving at hospital',h, 'with transport mode',j,'at time','t{}'.format(t))         
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')                        
print('------------------------------------------------INH(p,h,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for h in instance.h:
        for t in instance.t:
            if value(instance.INH[p,h,t]) == 1:
                print('Therapy',p,'arriving at hospital',h,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------CTM(p)------------------------------------------------------------------') 
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    print('Total manufacturing cost','of therapy' ,p,'is',value(instance.CTM[p]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')    
print('------------------------------------------------TTC(p)------------------------------------------------------------------') 
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    print('Total transport cost','of therapy' ,p,'is',value(instance.TTC[p]))

print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')    
print('------------------------------------------------TRT(p)------------------------------------------------------------------') 
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    print('Total return time','of therapy' ,p,'is',value(instance.TRT[p]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('Manufacturing facilities to be established')
for m in instance.m:
    if value(instance.E1[m])!=0:  
        print(m)
        
obj_val = value(instance.obj)
print('Total cost=',obj_val)
print('Average manufacturing cost per therapy', value(sum(instance.CTM[p] for p in instance.p))/len(instance.p))
print('Average transport cost per therapy', value(sum(instance.TTC[p] for p in instance.p))/len(instance.p))
print('Average QC cost per therapy',(10476+9312))
print('Average cost per therapy',obj_val/len(instance.p))
print('Average return time =',np.rint(value(instance.ATRT)))
print("CPU time:", t1_stop-t1_start)
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')





fig = plt.figure(figsize=(10,10))
Nc=4
Nm=6
Nh=4
leukapheresisY=np.linspace(0.1,0.9,Nc)
leukapheresisX=0.1*np.ones_like(leukapheresisY)
manufacturingY=np.linspace(0.1,0.9,Nm)
manufacturingX=0.5*np.ones_like(manufacturingY)
hospitalY=np.linspace(0.1,0.9,Nh)
hospitalX=0.9*np.ones_like(hospitalY)
for c in instance.c:
    for m in instance.m:
        if (value(instance.X1[c,m]))!=0:
            plt.plot([leukapheresisX[Nc-instance.c.ord(c)],manufacturingX[Nm-instance.m.ord(m)]],[leukapheresisY[Nc-instance.c.ord(c)],manufacturingY[Nm-instance.m.ord(m)]],color='darkmagenta')

for m in instance.m:
    for h in instance.h:
        if (value(instance.X2[m,h]))!=0:
            plt.plot([manufacturingX[Nm-instance.m.ord(m)],hospitalX[Nh-instance.h.ord(h)]],[manufacturingY[Nm-instance.m.ord(m)],hospitalY[Nh-instance.h.ord(h)]],color='darkmagenta')
            

for i in range(0,Nc):
    plt.scatter(leukapheresisX[i],leukapheresisY[i],s=250,color='teal')
    plt.text(leukapheresisX[Nc-i-1]-0.05,leukapheresisY[Nc-i-1],'C'+str(i+1),fontweight='bold')
for i in range(0,Nm):
    plt.scatter(manufacturingX[i],manufacturingY[i],s=250,color='teal')
    plt.text(manufacturingX[Nm-i-1]+0.03,manufacturingY[Nm-i-1],'M'+str(i+1),fontweight='bold')
for i in range(0,Nh):
    plt.scatter(hospitalX[i],hospitalY[i],s=250,color='teal')
    plt.text(hospitalX[Nh-i-1]+0.03,hospitalY[Nh-i-1],'H'+str(i+1),fontweight='bold')    
plt.axis('off')

fig.savefig('network.png')
