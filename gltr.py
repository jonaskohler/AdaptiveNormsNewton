import torch
import IPython
import exactTRsolver


def GLTR(grad, Hv, krylov_tol,tr_radius, M, exact_tol=1e-5,successful_flag=True,exact_solver="alg736",max_krylov_dim=1E6,verbose=0):
        """
        Minimize a constrained quadratic function using truncated (Steihaus Toint) Conjugate Gradients

        min b.Ts +1/2 s.THs s.t. ||s||<tr_radius (1)

        References
        ----------
        Algorithm 7.5.1: in Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.

        Parameters
        ----------
     
        gradient : 
            Gradient of f.(used as b in Eq. (1))
        Hv : callable Hv(grad,vec)
            Matrix-vector-product of matrix H and arbitrary vector vec. H must be symmetric!
        model_dim:
            dimensionality of the problem (len(b))
        krylov_tol: 
            tolerance for inexact solution (>0)
        tr_radius:
            constraint radius (see (1)). (>0)
        """
        def M_inv_v(vec):
            return 1/M*vec ### Assuming M=I
        def Mv(vec):
            return M*vec ### Assuming M=I

        grad = grad.squeeze()


        model_dim=grad.size(0)
        dtype=grad.type()
        subproblem_info = {}

        machine_precision=1E-20
        
        
        #if grad_norm < min(torch.sqrt(torch.norm(grad)) * torch.norm(grad),krylov_tol):
         #   return (torch.zeros_like(grad),0)

        # initialize
        s = torch.zeros_like(grad) #clones size and dtype
        g = grad.clone().detach()#g is the gradient
        v = M_inv_v(g)    #v is the pre-conditioned gradient
        p = -v.clone()
        k = 0
        model_decrease=0
        norm_s_squared=0
        
        alpha=0
        
        pMp=torch.dot(p,Mv(p)) 
        
        sMp=0### since s=0

        interior_flag=True

        alpha_k=[]
        beta_k=[]
        T_diag=[]
        T_offdiag=[]

        Q_list=[] #holds the cols of Q. A priori we don't know how many it is going to be
        #We create the matrix using torch.stack(Q_list,dim=1) only when needed in the very end.
        

        gv=torch.dot(g, v)
        grad_norm=torch.sqrt(gv) #=gamma_0

        if grad_norm==0:
            grad=grad+machine_precision*torch.Tensor(model_dim,1).normal_().type(dtype)

        while True:
            Bp=Hv(grad,p)
            pBp = torch.dot(p, Bp)+machine_precision
            
            if k>0: 
                sMp=beta*(sMp+alpha*pMp) #this needs a_k-1 and pmp_k-1!
            if k == 0:
                sigma = 1.0
            else:
                sigma = -torch.sign(alpha_k[k - 1]) * sigma
            alpha = gv / (pBp) #maybe add +machine precision here
            
            #### maybe switch to lanczos if pBp is mostly very small!!!

            ### GENERATE ORTOGHONAL BASIS (Lanczos) FROM CG INFO
            Q_list.append((sigma*v/torch.sqrt(gv)).squeeze()) #append as vectors

        

            alpha_k.append(alpha)

            #Build T_k. appending is O(1) and thus cheaper than concatenating
            

            #if k==0:
            #    T_diag.append(torch.tensor([1 / alpha]).type(dtype))
            #elif k>0:
                #note that the counter k of beta lags one behind. So beta here, is beta_k-1
            #    T_diag.append((T_diag,torch.tensor([1. / alpha+beta/alpha_k[k-1]]).type(dtype)))
            #    T_offdiag.append((T_offdiag, torch.tensor(torch.sqrt(beta)/torch.abs(alpha_k[k-1])).type(dtype)))
 

 # b) Create T for Lanczos Model (inefficient way)

            if k == 0:
                T_diag.append(1 / alpha.item())
            else:
                T_diag.append((1 / alpha+ beta/ alpha_k[k - 1]).item())
                T_offdiag.append((torch.sqrt(beta) / torch.abs(alpha_k[k - 1])).item())
            ### COMPUTE TRIAL STEP LENGTH ||s+alpha p||_M^2 

            if k>0:
                pMp=beta**2*pMp+gv
            norm_s_squared_old=norm_s_squared
            norm_s_squared=norm_s_squared+2*alpha*sMp+alpha**2*pMp #=(s+alpha p).TM(s+alpha p)

            ### TEST FOR BOUNDARY SOLUTION
            if interior_flag==True and (alpha <=0  or torch.sqrt(norm_s_squared)>=tr_radius):
                interior_flag=False
                if verbose  >0:
                    print('switching to lanczos')
                subproblem_info['info'] = 'switching to lanczos'

            ### a) RUN CG UPDATE
            if interior_flag: 
                if k==0:
                    lambda_k=0
                s = s + alpha * p
                model_decrease+=-1/2*alpha*gv
            else:
            ### or b) RUN LANCZOS
                e_1=torch.zeros(k+1).type(dtype)
                e_1[0]=1.0
                g_lanczos=grad_norm*e_1 #grad_norm=gamma

                if k==0: #one_d probem. solve this by hand to avoid hustle with Toffdiag etc.
                    if T_diag[0]>0: ## interior solution
                        h=-grad_norm/T_diag[0]
                        lambda_k=0 

                    elif grad_norm==0 and T_diag[0]==0: ##local minimum???
                        h=0
                        lambda_k=0
                        if verbose>0:
                            print('something weird might be happening right now')
                    else:
                        h=-tr_radius ## border solution. always "left" since grad norm is always non-negative (increasing function)
                        lambda_k=torch.abs((tr_radius/grad_norm)-T_diag[0])
                    model_decrease=h*grad_norm+1/2*T_diag[0]*h**2

                else:
                    if exact_solver=="alg736":
                        #IPython.embed()

                        h,lambda_k=exactTRsolver.alg736(g_lanczos.detach().numpy(),T_diag,T_offdiag,tr_radius,exact_tol,successful_flag,lambda_k)
                        h=torch.from_numpy(h).type(dtype).squeeze()
                       # lambda_k=torch.from_numpy(lambda_k).type(dtype)
                    elif exact_solver=="alg736_restarted":
                        h,lambda_k=exactTRsolver.alg736_restarted(g_lanczos,T_diag,T_offdiag,tr_radius,exact_tol,successful_flag,lambda_k)
                        h=torch.from_numpy(h).type(dtype).squeeze()
                        #lambda_k=torch.from_numpy(lambda_k).type(dtype)
                    else:    
                        if exact_solver=="alg734":
                            T=torch.diagflat(torch.Tensor(T_diag).type(dtype))
                            if k>0:
                                T+=torch.diagflat(torch.Tensor(T_offdiag).type(dtype),offset=1)
                                T+=torch.diagflat(torch.Tensor(T_offdiag).type(dtype),offset=-1)
                            h,lambda_k=exactTRsolver.alg734(g_lanczos,T, tr_radius, exact_tol,successful_flag,lambda_k)
                        elif exact_solver=="alg734_cpu":
                            h,lambda_k=exactTRsolver.alg734_cpu(g_lanczos,T_diag,T_offdiag,tr_radius,exact_tol,successful_flag,lambda_k)
                            h=torch.from_numpy(h).type(dtype).squeeze()

                            #lambda_k=torch.from_numpy(lambda_k).type(dtype)
                        else:  
                            raise ValueError('specified exact solver is not available') 


                # call exact solver with g_lanczos and T to get back h
                    #import IPython; IPython.embed(); exit(1)

                    model_decrease=1/2*(grad_norm*h[0]-lambda_k*tr_radius**2) #overall decrease, thus no += !!
                    #print(model_decrease)
                #h=
            g = g + alpha * Bp
            v=M_inv_v(g)
            gv_old=gv
            gv=torch.dot(g, v)


            ###### CONVERGENCE TEST ######
            if interior_flag==True:
                #test convergence using gv

                if torch.sqrt(gv) <= min(torch.sqrt(grad_norm),krylov_tol)*grad_norm or k==max_krylov_dim:  #giving CG some slack for numerical inaccuracies
                #if k==int(2*model_dim):
                   # import IPython; IPython.embed(); exit(1)  
                    subproblem_info['steps'] = k
                    subproblem_info['info'] = "lol"

                    del Q_list
                    del T_diag
                    del T_offdiag
                    return s,torch.abs(model_decrease),subproblem_info
            else:
                e_k=torch.zeros(k+1).type(dtype)
                e_k[k]=1.0
                if k>0:
                    #test convergence using lanczos stuff.
                    # if k > 1:
                    #     exit()
                    if torch.abs(h.squeeze()[-1])*torch.sqrt(gv) <= min(grad_norm,krylov_tol)*grad_norm or k==max_krylov_dim:
                    #if k==model_dim:
                        subproblem_info['steps'] = k
                        subproblem_info['info'] = "lol"
                        final_step=torch.mv(torch.stack(Q_list,dim=1),h)
                        del Q_list
                        del T_diag
                        del T_offdiag
                        return final_step,torch.abs(model_decrease),subproblem_info


    

            beta = gv / gv_old #new over old (computed above)
            p = -v + beta * p
            # update iterates
            k = k + 1

