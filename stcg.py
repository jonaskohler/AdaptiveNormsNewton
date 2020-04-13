import torch
import IPython
import time
#import numpy as np


def truncatedCG(grad, Hv, krylov_tol,tr_radius,M, epoch, epochs_1storder=-1, max_krylov_dim=1E6, my_preconditioner="uniform",verbose=0,precon_stopping_criterion=False,**kwargs):
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

        M is a vector of dx1 that contains the diagonal preconditioner. 
        Pass M=torch.ones(d)  for spherical trust regions!
        """

        def Mv(vec):
            return (M*vec).data ### Assuming M is diagonal

        def M_inv_v(vec):
            return (1/M*vec).data ### Assuming M is diagonal

        def M_norm_squared(vec):
            return torch.sum(vec**2 * M) # Assuming M is diagonal

        def boundary_point_along_p(pMp, sMp, norm_s_squared_old, tr_radius):
            # Return point on boundary. Compute root according to formula 7.5.5
            c = norm_s_squared_old - tr_radius ** 2

            sqrt_discriminant = torch.sqrt(sMp**2 - pMp * c)
            kappa = (-sMp + sqrt_discriminant) / pMp

            # note that m(x+s)-m(x) for STCG is given on p793 as
            model_decrease = -kappa * gv + 1 / 2 * kappa**2 * pBp

            return model_decrease, kappa

        if len(list(grad.size())) > 1:
            # Gradient vector needs to be flat
            grad = grad.squeeze()


        learning_rate=0.01   #####################################################################################################

        model_dim = grad.size(0)
        subproblem_info = {}
        tr_radius = min(tr_radius, 1E10)

        machine_precision = 1e-10
        grad_norm = torch.norm(grad)
        #alt: grad_norm=torch.sqrt(gv)  <- see below!


        if grad_norm == 0:
            raise AttributeError
            # dtype = grad.type()
            # grad = grad + machine_precision * torch.Tensor(model_dim).normal_().type(dtype)


        # initialize
        s = torch.zeros_like(grad) #this clones sizes and dtype!
        #g = grad.clone().detach()  #g is the gradient
        g=grad.data.clone().detach()
        v = M_inv_v(g)    #v is the pre-conditioned gradient
        p = -v.clone()
        k = 0

        model_decrease = 0
        norm_s_squared = 0
        alpha = 0
        pMp = p @ Mv(p)
        sMp = 0 # since s=0
        gv = g @ v
        # grad_norm=torch.sqrt(gv) # preconditioned gradient norm

        ### s.Tg + (1-1/t) * 1/2*s.THs
        #factor=1-1/(i+1)**(1/2)  ## CG now are minimizes the model g.Ts + factor*1/2*s.T.Hs

        # i=torch.Tensor([i]).type(grad.type())
        # factor=torch.sigmoid(0.03*(i-5000))


        factor = 1#0 if epoch <= epochs_1storder else 1

        # factor = 1. # fully trust region approach
        while True:
            if factor==0: #now we have a fully linear model
             
                s_next = s - learning_rate * M_inv_v(g) #################################################################################
                #s_next = s - learning_rate * g
                model_decrease = abs(torch.dot(s_next, g).item())

                subproblem_info["info"] = "did_sgd"  # ??? no longer needed?
                subproblem_info["steps"] = -1

                return s_next, model_decrease, subproblem_info
      
            else:
                Bp = Hv(grad,p)
                pBp = p @ Bp # <p, Hp> ############################################################################################

                pBp=factor*pBp
                if k > 0: # Efficient way to compute sMp (7.5.6)
                    sMp = beta * (sMp + alpha * pMp) #this needs a_k-1 and pmp_k-1!
                alpha = gv / pBp #maybe add +machine precision here

                #computing norm of trial step s_k+1. See 7.5.5-7.5.7 in Conn book. cheaper since no need for repeated Mvec products
                if k > 0:
                    pMp = beta**2 * pMp + gv # 7.5.7

                norm_s_squared_old = norm_s_squared
                norm_s_squared = norm_s_squared + 2 * alpha * sMp + alpha**2 * pMp

                if pBp <= 0:
                    # Negative curvature found. Return point on boundary. Compute root according to formula 7.5.5
                    additional_model_decrease, kappa = boundary_point_along_p(pMp, sMp, norm_s_squared_old, tr_radius)

                    # note that m(x+s)-m(x) for STCG is given on p793 as
                    model_decrease += additional_model_decrease

                    if verbose > 0:
                        print('negative curvature found')
                    subproblem_info["info"] = "negative curvature found"
                    subproblem_info["steps"] = k

                    return s + kappa * p, abs(model_decrease.item()), subproblem_info

                s_next = s + alpha * p

                if torch.sqrt(M_norm_squared(s_next)) >= tr_radius:
                    # Step outside the boundary. Return point on boundary.
                    additional_model_decrease, kappa = boundary_point_along_p(pMp, sMp, norm_s_squared_old, tr_radius)

                    # note that m(x+s)-m(x) for STCG is given on p793 as
                    model_decrease += additional_model_decrease

                    if verbose > 0:
                        print('bumped into boundary')
                    subproblem_info["info"] = "bumped into boundary"
                    subproblem_info["steps"] = k

                    return s + kappa * p, abs(model_decrease.item()), subproblem_info

                # Proposed step lies within trust region.
                model_decrease_old=model_decrease
                model_decrease -= (1/2 * alpha * gv).item()

                g += alpha * factor *Bp    ###############################################################################################
                v = M_inv_v(g)

                ## should we use the M norm of g for termination?? (I am giving some slack in terms of model_dim for numerical reasons)
                #if torch.sqrt(gv) <= min(torch.sqrt(grad_norm),krylov_tol)*grad_norm or k==min(int(2*model_dim), 25):
               

                if precon_stopping_criterion: #based on model decrease
                    if k>0:
                        if model_decrease_old==0:
                            subproblem_info["info"] = "{} CG_iterations".format(k)
                            subproblem_info["steps"] = k

                            return s_next, abs(model_decrease), subproblem_info
                        elif  (model_decrease-model_decrease_old)/(model_decrease_old) <= min(torch.sqrt(grad_norm).item(),krylov_tol) or k==max_krylov_dim:
                            if verbose > 0:
                                print('CG_iterations', k)
                            subproblem_info["info"] = "{} CG_iterations".format(k)
                            subproblem_info["steps"] = k

                            return s_next, abs(model_decrease), subproblem_info

                else:
                    if torch.norm(g).item() <= min(torch.sqrt(grad_norm).item(),krylov_tol)*grad_norm.item() or k==max_krylov_dim:
                        if verbose > 0:
                            print('CG_iterations', k)
                        subproblem_info["info"] = "{} CG_iterations".format(k)
                        subproblem_info["steps"] = k

                        return s_next, abs(model_decrease), subproblem_info

                gv_old = gv
                gv = g @ v
                beta = gv / gv_old #new over old (computed above)
                p = -v + beta * p
                # update iterates
                s = s_next
                k = k + 1

