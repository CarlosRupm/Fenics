#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fenics import *
from mshr import *
import numpy as np
from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
from petsc4py import PETSc
import math

#Unsteady Navier-Stokes
T = 300
n_sps=100
num_steps = int(T)*n_sps
dt = T / num_steps
print(num_steps)
mu = 0.009 #dynamic viscosity
y_h= 40
rho = 1  # density
U0=1     #initial horizontal velocity
L=y_h*2   # height
Re=100 #Reynold's number
D=1
mu=rho*U0*D/Re
Rer=round(Re,2)


sup=0


# In[2]:


# Create mesh

channel = Rectangle(Point(-40.0, -y_h), Point(50.0, y_h))


cylinder = Circle(Point(0.0, 0.0), 0.5,320)
   
domain = channel - cylinder
mesh = generate_mesh(domain, 30)#35

#Refine mesh code
cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())

for c in cells(mesh):
    if c.midpoint().y() < 9 and c.midpoint().y() > -9 and c.midpoint().x() < 24 and c.midpoint().x() > -9:
                cell_markers[c] = True
    else:
                cell_markers[c] = False

mesh = refine(mesh, cell_markers)

cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())

for c in cells(mesh):
    
    if c.midpoint().y() < 3 and c.midpoint().y() > -3 and c.midpoint().x() < 12 and c.midpoint().x() > -3:
                
                cell_markers[c] = True
    else:
                cell_markers[c] = False
    

mesh = refine(mesh, cell_markers)

mesh_file=XDMFFile('mesh.xdmf')
mesh_file.write(mesh)

#Injection zone       

theta_d=22.5
theta_ini=22.5*math.pi/180
ampli=0.1

max_x=0.5*math.cos(theta_ini+ampli)
min_x=0.5*math.cos(theta_ini-ampli)
max_y=0.5*math.sin(theta_ini+ampli)
min_y=0.5*math.sin(theta_ini-ampli)
print(max_x,max_y,min_x,min_y)
print(max_x*max_x+max_y*max_y)


#Define ds for several boundarys

class BoundaryCy(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]>-0.55 and x[0]<0.55 and x[1]>-0.55 and x[1]<0.55
    
class BoundaryOut(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],50) 

class BoundaryWll(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],y_h) or on_boundary and near(x[1],-y_h)

class BoundaryCy_inj(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and x[0]<min_x and x[0]>max_x and x[1]>min_y and x[1]<max_y) or (on_boundary and x[0]<min_x and x[0]>max_x and x[1]<-min_y and x[1]>-max_y) 
    
class BoundaryCy_Noinj(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and x[0]>min_x or x[0]<max_x and x[1]<min_y or x[1]>max_y) or (on_boundary and x[0]>min_x or x[0]<max_x and x[1]>-min_y or x[1]<-max_y) 
        
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)  
boundary_markers.set_all(False)
cy = BoundaryCy()
cy.mark(boundary_markers, True)

boundary_markers_wall = MeshFunction("size_t", mesh, mesh.topology().dim()-1)  
boundary_markers_wall.set_all(False)
#boundary_markers_out = MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
wll = BoundaryWll()
wll.mark(boundary_markers_wall, True)


boundary_markers_inj = MeshFunction("size_t", mesh, mesh.topology().dim()-1)  
boundary_markers_inj.set_all(False)
cy_inj = BoundaryCy_inj()
cy_inj.mark(boundary_markers_inj, True)

boundary_markers_Noinj = MeshFunction("size_t", mesh, mesh.topology().dim()-1)  
boundary_markers_Noinj.set_all(False)
cy_Noinj = BoundaryCy_Noinj()
cy_Noinj.mark(boundary_markers_Noinj, True)

boundary_markers_outflow = MeshFunction("size_t", mesh, mesh.topology().dim()-1)  
boundary_markers_outflow.set_all(False)
out = BoundaryOut()
out.mark(boundary_markers_outflow, True)

dsc=Measure('ds', domain=mesh, subdomain_data=boundary_markers, subdomain_id=True)
dsci=Measure('ds', domain=mesh, subdomain_data=boundary_markers_inj, subdomain_id=True)
dscni=Measure('ds', domain=mesh, subdomain_data=boundary_markers_Noinj, subdomain_id=True)
dsw=Measure('ds', domain=mesh, subdomain_data=boundary_markers_wall, subdomain_id=True)  
dsout=Measure('ds', domain=mesh, subdomain_data=boundary_markers_outflow, subdomain_id=True)  
#dso=Measure('ds', domain=mesh, subdomain_data=boundary_markers_out)   



# Define function spaces

#Product of Function Spaces
P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)
Q = FunctionSpace(mesh, 'CG', 1)
V = VectorFunctionSpace(mesh, 'CG', 1)



#normal vector function
un = TrialFunction(V)
vn = TestFunction(V)
n = FacetNormal(mesh)
an = inner(un,vn)*dsci
ln = inner(n, vn)*dsci
An = assemble(an, keep_diagonal=True)
Ln = assemble(ln)

An.ident_zeros()
nh_inj = Function(V)

solve(An, nh_inj.vector(), Ln)
#plot(nh_inj)

#normal vector function
un1 = TrialFunction(V)
vn1 = TestFunction(V)
an1 = inner(un1,vn1)*dsc
ln1 = inner(n, vn1)*dsc
An1 = assemble(an1, keep_diagonal=True)
Ln1 = assemble(ln1)

An1.ident_zeros()
nh = Function(V)

solve(An1, nh.vector(), Ln1)
#plot(nh)

#normal vector function
unninj = TrialFunction(V)
vnninj = TestFunction(V)
anninj = inner(unninj,vnninj)*dscni
lnninj = inner(n, vnninj)*dscni
Anninj = assemble(anninj, keep_diagonal=True)
Lnninj = assemble(lnninj)

Anninj.ident_zeros()
nhninj = Function(V)

solve(Anninj, nhninj.vector(), Lnninj)
plot(nhninj)

xdmffile_inj = XDMFFile('../Plot/Base/inj.xdmf')
xdmffile_inj.parameters["flush_output"]=True

xdmffile_inj.write(nh_inj)

# Define boundaries
inflow= 'near(x[0], -40)'
outflow= 'near(x[0], 50)'
walls= 'near(x[1], '+str(-y_h)+') || near(x[1], '+str(y_h)+' ) '
cylinder= 'on_boundary && x[0]>-0.55 && x[0]<0.55 && x[1]>-0.55 && x[1]<0.55'

# Define boundary conditions
inflow_profile = ('1', '0')

bcu_inflowp = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), inflow)
bcu_wallsp = DirichletBC(W.sub(0), Constant((1, 0)), walls)
bcu_cylinderp = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
bcp_outflowp = DirichletBC(W.sub(1), Constant(0), outflow)
bcs = [bcu_inflowp, bcu_wallsp, bcu_cylinderp, bcp_outflowp]

#w = Function(W)
#u,p = split(w) 
#v, q=TestFunctions(W)
#f = Constant((0, 0))   

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((1, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow , bcu_walls,bcu_cylinder]
bcp = [bcp_outflow]

coor = mesh.coordinates()
dof = coor.shape[0]

xp=6
yp=0.0
xinf=0
ip=0
ipc=[]
ipinf=0
points_refine=np.zeros(dof)
#locate point and cylinder
for v in vertices(mesh):
    x = v.point().x()
    y = v.point().y()
    if (abs(xp-x)<0.3 and abs(yp-y)<0.3):
        print(x,y)
        ip=v.index()
        
    if (x<12 and y<3 and x>-3 and y>-3):
        points_refine[v.index()]=1
        #print(v.index())
print(dof)
npr=len(points_refine)
#print(points_refine)


# In[3]:


#Define trial and test functions
u= TrialFunction(V)
u_mean=Function(V)
v= TestFunction(V)
#v, q= TestFunctions(W)
p=TrialFunction(Q)
q= TestFunction(Q)
#u,p=split(w)
# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_ = Function(V)
#u_,p_=split(w_)
p_n = Function(Q)
p_ = Function(Q)
u_aux = Function(V)
# Define expressions used in variational forms
U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
u_mean=project(f,V)
number_of_points=u_mean.vector()[:].shape[0]
print(number_of_points)
k = Constant(dt)
mu = Constant(mu)
prueba=Function(Q)
zero=Constant(0)
prueba=project(zero,Q)
#for i in range(len(points_refine)):
#    prueba.compute_vertex_values()[points_refine[i]]=1.0

#for v in vertices(mesh):
#    x = v.point().x()
#    y = v.point().y()
#    if (x<12 and y<3 and x>-3 and y>-3):
#        prueba.vector()[v.index()]=1.0


#Para guardar solution antes de que el vortice despierte
u_vor=[]
p_vor=[]


if( sup == 0):
    u_vor=np.load('u_vor.npy')
    p_vor=np.load('p_vor.npy')
    u_n.vector()[:]=u_vor[:]
    p_n.vector()[:]=p_vor[:]

      
#prueba.vector()[:] = points_refine[Q.dofmap().dofs()]
dim = Q.dofmap().index_map().size(IndexMap.MapSize.OWNED)
print(dim)
prueba.vector().set_local(points_refine[dof_to_vertex_map(Q)[:dim]].copy())
plot(prueba)    


# In[4]:


#Definition of time-step storage
#Vortex time
t_vortex=90
n_vortex=int(t_vortex*n_sps)
nsnap=int((num_steps-n_vortex))
print(nsnap)
npr2=dof*2

u_steps = np.zeros((npr2,nsnap))
p_steps = np.zeros((dof,nsnap))
uad_steps = np.zeros((npr2,nsnap))
pad_steps = np.zeros((dof,nsnap))


# In[5]:


# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))
# Define variational problem for step 1

#vtkfile = File('Pv/flow_n.pvd')



F1 = rho*dot((u - u_n) / k, v)*dx     + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx     + inner(sigma(U, p_n), epsilon(v))*dx     + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds     - dot(f, v)*dx
L= 0
a1 = lhs(F1)
L1 = rhs(F1)
# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]
# Create XDMF files for visualization output
xdmffile_u = XDMFFile('../Plot/Base/velocity_500.xdmf')
xdmffile_u.parameters["flush_output"]=True
xdmffile_p = XDMFFile('../Plot/Base/pressure_500.xdmf')
xdmffile_p.parameters["flush_output"]=True
# Create time series (for use in reaction_system.py)
#timeseries_u = TimeSeries('velocity_series')
#timeseries_p = TimeSeries('navier_stokes_cylinder/pressure_series')
# Save mesh to file (for use in reaction_system.py)
#File('navier_stokes_cylinder/cylinder.xml.gz') << mesh
# Create progress bar
progress = Progress('Time-stepping')
#set_log_level(PROGRESS)
# Time-stepping
t = 0
tp=[]
upx=[]
upy=[]
upm=[]
pv=[]
Drag_t=[]

foto=0


# In[6]:


nm=0
Drag1=0
Drag2=0
Lift=0
ix = Constant((0,1))

if(sup==0):
    t=dt*n_vortex
    a=n_vortex
for n_S in range(a,num_steps):
    # Update current time
    if (n_S%500==0):
        print(n_S)
    t += dt
    i=0
    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')
    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')
    # Plot solution
    #plot(u_, title='Velocity')
    #plot(p_, title='Pressure')
    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t) 
    #vtkfile<< u_
    # Save nodal values to file
    #timeseries_u.store(u_.vector(), t)
    #timeseries_p.store(p_.vector(), t)
    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    vertex_values_uv = u_.compute_vertex_values()
    vertex_values_pv = p_.compute_vertex_values()
    if (n_S > n_vortex):
        u_steps[:,nm]=u_.vector()[:]
        p_steps[:,nm]=p_.vector()[:]
        nm=nm+1
        upx.append(vertex_values_uv[ip])
        upy.append(vertex_values_uv[ip+dof])
        tp.append(t)
        Drag1= Drag1 + assemble(dot(sigma(u_,p_),n)[0]*dsc)
        Drag_t.append(assemble(dot(sigma(u_,p_),n)[0]*dsc))
        Lift=Lift +  assemble(dot(sigma(u_,p_),n)[1]*dsc)
        #Drag2= Drag2+ assemble(dot(sigma(u_,p_),n)[1]*dsci)
    # Update progress bar
    #progress.update(t / T)
# Hold plot


# In[7]:


Drag1=Drag1/nsnap
Drag2=Drag2/nsnap
print(Drag1)
print(Drag1-Drag2)
Lift=Lift/nsnap
print(Lift)
tp_num=np.array(tp)
Drag_num=np.array(Drag_t)
np.save("tstep_500.npy",tp_num)
np.save("Drag_t_inicial_500.npy",Drag_num)
np.save("u_base_500.npy",u_steps)
np.save("p_base_500.npy",p_steps)


# In[8]:


plt.figure(figsize=(10,10))
plt.plot(tp,upy)
plt.grid()
plt.xlabel('Time  (s)', fontsize=20)
plt.ylabel('Velocity (m/s)', fontsize=20)
plt.title("Horizotal velocity y, Re~"+str(Rer)+" at x~"+str(xp)+"  y~"+str(yp),fontsize=20)
plt.savefig('y_velocity_'+str(dof)+'.png')


Utt=np.fft.fft(upy)/nsnap
freq = np.fft.fftfreq(nsnap,dt)


# In[9]:


minor_ticks_top=np.linspace(-0.027,0.027,42)
minor_ticks_x=np.linspace(-10,10,30)
plt.figure(figsize=(10,10))
#plt.axes().set_yticks(minor_ticks_top)
#plt.axes().set_xticks(minor_ticks_x)
plt.grid(True)
plt.vlines(freq ,0, Utt.imag)
print(np.min(Utt.imag))
print(np.argmax(Utt.imag))
strouhal=freq[np.argmax(Utt.imag)]
print(strouhal)
Utt2=np.copy(Utt)
Utt2[np.argmax(Utt.imag)]=0
print(np.max(Utt2.imag))
print(np.argmax(Utt2.imag))
print(freq[np.argmax(Utt2.imag)])


# In[10]:


f = open ('mesh_convergence.txt','a')
f.write(str(dof)+' '+ str(Drag1)+' '+str(strouhal))
f.close()


# In[1]:


bcu_inflowa = DirichletBC(W.sub(0), Constant((0, 0)), inflow)
bcu_wallsa = DirichletBC(W.sub(0), Constant((0, 0)), walls)
bcu_cylindera = DirichletBC(W.sub(0), Constant((2, 0)), cylinder)
bcp_outflowa = DirichletBC(W.sub(1), Constant(0), outflow)
bcsa = [bcu_inflowa, bcu_wallsa, bcu_cylindera, bcp_outflowa]
Vp = VectorFunctionSpace(mesh, 'CG', 2)

wa = Function(W)
ua,pa = split(wa) 
va, qa=TestFunctions(W)
wan = Function(W)
uan,pan = split(wan)
ud_ = Function(V)
ua_ = Function(Vp)
pa_ = Function(Q)
ud_1 = Function(Vp)
ua_v = Function(V)
wd = Function(W)
ud,pd = split(wd)


adjoint_solution = XDMFFile("../Plot/Adjoint/a_solution_500.xdmf")
adjoint_solution.parameters["flush_output"]=True
direct_solution = XDMFFile("../Plot/Adjoint/d_solution_500.xdmf")
direct_solution.parameters["flush_output"]=True

vec_aux=[]
print(nm)
print(nsnap)
t=0
for i in range(nm):
    if(i%100==0):
        print(i)
    
    vec_aux[:]=u_steps[:,nm-i-1]
    ud_.vector()[:]=vec_aux[:]
    ud_1=project(ud_,Vp)
    assign(wd.sub(0),ud_1)
    direct_solution.write(wd,t)
     
    t += dt
    aa=(1/k)*rho*dot(ua-uan,va)*dx +rho*dot(dot(ua,nabla_grad(ud).T),va)*dx-rho*dot(dot(ud,nabla_grad(ua)),va)*dx +  inner(sigma(ua,-pa),epsilon(va))*dx+ dot(-pa*n, va)*dsout - dot(mu*nabla_grad(ua)*n, va)*dsout+ dot(dot(ud,n)*ua,va)*dsout + qa*div(ua)*dx 
    
    solve(aa==0,wa,bcsa)
    assign(ua_,wa.sub(0))
    ua_v=project(ua_,V)
    assign(pa_,wa.sub(1))
    pa_v=project(pa_,Q)
    uad_steps[:,nm-i-1]=ua_v.vector()[:]
    pad_steps[:,nm-i-1]=pa_v.vector()[:]
    adjoint_solution.write(wa,t)
    wan.assign(wa)
    


# In[ ]:


t=0
sensi_g_inj_solution=XDMFFile('../Plot/Sensitivity_drag/sensi_g_inj_500.xdmf')
sensi_g_inj_solution.parameters["flush_output"]=True
sensi_g_solution=XDMFFile('../Plot/Sensitivity_drag/sensi_g_500.xdmf')
sensi_g_solution.parameters["flush_output"]=True
comp_solution=XDMFFile('../Plot/Sensitivity_drag/comp_g_500.xdmf')
comp_solution.parameters["flush_output"]=True
sensi_shape_solution=XDMFFile('../Plot/Sensitivity_drag/sensi_shape_500.xdmf')
sensi_shape_solution.parameters["flush_output"]=True
#sensi_f_solution=File("../Plot/Sensitivity_drag/sensi_f.xdmf")
#sensi_f_solution.parameters["flush_output"]=True
ug=Function(V)
pg=Function(Q)
def_mean=Function(V)
sensi_g=Function(V)
sensi_g_inj=Function(V)
sensi_shape=Function(V)
#Comparar las dos sensibilidades, la total y la restringida
comp_g = Function(V)
vec_auxa=[]
vec_auxp=[]
sen_numpy=np.zeros((npr2,nsnap-1))
senshape_numpy=np.zeros((npr2,nsnap))
contador=0
for i in range(nsnap-1):
    if(i%100==0):
        print(i)
    t += dt
    vec_aux[:]=u_steps[:,i]
    ud_.vector()[:]=vec_aux[:]
    vec_auxa[:]=uad_steps[:,i]
    vec_auxp[:]=pad_steps[:,i]
    ug.vector()[:]=vec_auxa[:]
    pg.vector()[:]=vec_auxp[:]
    
    
    theo_sensi=dot(sigma(ug,-pg),nh)
    project(theo_sensi,V,function=sensi_g)
    
    theo_sensi_inj=dot(sigma(ug,-pg),nh_inj)
    project(theo_sensi_inj,V,function=sensi_g_inj)
    
    theo_sensi_shape=dot(sensi_g,dot(-nh,nabla_grad(ud_)))*nh
    project(theo_sensi_shape,V,function=sensi_shape)
    
    def_mean.vector()[:]=def_mean.vector()[:]+sensi_shape.vector()[:]
    contador= contador+1
    
    
    comp_g.vector()[:]=sensi_g.vector()[:]-sensi_g_inj.vector()[:]
    sensi_g_inj_solution.write(sensi_g_inj, t)
    sensi_g_solution.write(sensi_g, t)
    sensi_shape_solution.write(sensi_shape, t)
    comp_solution.write(comp_g,t)
    sen_numpy[:,nsnap-2-i]=sensi_g.vector()[:]
    senshape_numpy[:,nsnap-2-i]=sensi_shape.vector()[:]
    

def_mean.vector()[:]=  def_mean.vector()[:]/contador
senshape_numpy[:,nsnap-1]=def_mean.vector()[:]


# In[ ]:


np.save('sensi_500.npy',sen_numpy)
np.save('sensi_shape_500.npy',senshape_numpy) 


# In[ ]:


sensi_g_solution=XDMFFile('../Plot/Sensitivity_drag/sensi_g_500.xdmf')
sensi_g_solution.parameters["flush_output"]=True
ug=Function(V)
pg=Function(Q)
sensi_g=Function(V)
vec_auxa=[]
vec_auxp=[]
for i in range(nsnap-1):
    t += dt
    vec_auxa[:]=uad_steps[:,i]
    vec_auxp[:]=pad_steps[:,i]
    ug.vector()[:]=vec_auxa[:]
    pg.vector()[:]=vec_auxp[:]
    theo_sensi=dot(sigma(ug,-pg),nh)
    project(theo_sensi,V,function=sensi_g)
    sensi_g_solution.write(sensi_g, t)


# In[21]:


vec_mean=[]
umean = Function(V)
for i in range (nm):
    vec_mean[:]=u_steps[:,nm-i-1]
    umean.vector()[:]=umean.vector()[:]+vec_mean[:]
umean.vector()[:]=umean.vector()[:]/nm
plot(umean)


# In[22]:


mean_solution=XDMFFile('../Plot/Base/mean.xdmf')
mean_solution.parameters["flush_output"]=True
mean_solution.write(umean)


# In[ ]:




