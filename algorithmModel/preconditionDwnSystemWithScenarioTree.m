function [ dwnOptimModelPrecond, detailsPrecond] = preconditionDwnSystemWithScenarioTree(dwnOptimModel, ...
    treeData, dwnOptimCost, optionPrecondition)
% This function calculates the precondition for dual proximal method applied to the
%   DWN for the complete scenario tree
%
% INPUT
%   dwnOptimModel  :
%   treeData       :
%   dwnOptimCost   :
%   apg_opts       :
%
% OUTPUT
%   dwnOptimModelPrecond  :
%   precondDetail :
%

nx = dwnOptimModel.nx;
nu = dwnOptimModel.nu;
nz = 2*nx+nu;
nStage = size(treeData.stage,1);
nPred = dwnOptimModel.Np;

dwnOptimModelTemp.A = dwnOptimModel.A;
dwnOptimModelTemp.B = dwnOptimModel.B;
dwnOptimModelTemp.nx = dwnOptimModel.nx;
dwnOptimModelTemp.nu = dwnOptimModel.nu;
dwnOptimModelTemp.Np = dwnOptimModel.Np;

dwnOptimModelTemp.F = cell(nPred,1);
dwnOptimModelTemp.G = cell(nPred,1);
dwnOptimModelTemp.cell = 1;

treeDataTemp.stage= (0:nPred-1)';
treeDataTemp.leaves = nPred;
treeDataTemp.prob = ones(nPred,1);

for i=1:nPred
    dwnOptimModelTemp.F{i} = dwnOptimModel.F{1};
    dwnOptimModelTemp.G{i} = dwnOptimModel.G{1};
end
dwnOptimModelTemp.umin = kron(ones(nPred,1),dwnOptimModel.umin(1:nu,1));
dwnOptimModelTemp.umax = kron(ones(nPred,1),dwnOptimModel.umax(1:nu,1));

dwnOptimModelTemp.xmin = kron(ones(nPred,1),dwnOptimModel.xmin(1:nx,1));
dwnOptimModelTemp.xmax = kron(ones(nPred,1),dwnOptimModel.xmax(1:nx,1));
dwnOptimModelTemp.xs = kron(ones(nPred,1),dwnOptimModel.xs(1:nx,1));

[dwnDualHessianTemp, detailDualHessianTemp] = calculateDwnDualHessian(dwnOptimModelTemp,...
    treeDataTemp, dwnOptimCost, optionPrecondition);
detailsPrecond.norm_act = detailDualHessianTemp.norm;

if(optionPrecondition.exact_prcnd)
    [dwnOptimModelTempPrecond, detailsPrecond.pre_cnd] = precond_modify_sdp_box(dwnOptimModelTemp,dwnDualHessianTemp);
    
    [~, detailsDwnHessianPrecondTemp] = dual_hessian_modfy_box(dwnOptimModelTempPrecond,treeDataTemp,dwnOptimCost,optionPrecondition);
    
    detailsPrecond.norm = detailsDwnHessianPrecondTemp.norm;
    
    prob=sqrt(treeData.prob);
    
    dwnOptimModelPrecond=dwnOptimModel;
    
    for j=1:nStage
        dwnOptimModelPrecond.G{j,1}(nx+1:end,:)=prob(j)*detailsPrecond.pre_cnd.Eg1*dwnOptimModel.G{j,1}(nx+1:end,:);
        dwnOptimModelPrecond.umax((j-1)*nu+1:j*nu,1)=prob(j)*detailsPrecond.pre_cnd.Eg1*dwnOptimModel.umax((j-1)*nu+1:j*nu,1);
        dwnOptimModelPrecond.umin((j-1)*nu+1:j*nu,1)=prob(j)*detailsPrecond.pre_cnd.Eg1*dwnOptimModel.umin((j-1)*nu+1:j*nu,1);
        
        dwnOptimModelPrecond.F{j,1}(1:nx,:)=prob(j)*detailsPrecond.pre_cnd.Ef1*dwnOptimModel.F{j,1}(1:nx,:);
        dwnOptimModelPrecond.xmax((j-1)*nx+1:j*nx,1)=prob(j)*detailsPrecond.pre_cnd.Ef1*dwnOptimModel.xmax((j-1)*nx+1:j*nx,1);
        dwnOptimModelPrecond.xmin((j-1)*nx+1:j*nx,1)=prob(j)*detailsPrecond.pre_cnd.Ef1*dwnOptimModel.xmin((j-1)*nx+1:j*nx,1);
        dwnOptimModelPrecond.xs((j-1)*nx+1:j*nx,1)=prob(j)*detailsPrecond.pre_cnd.Ef1*dwnOptimModel.xs((j-1)*nx+1:j*nx,1);
    end
else
    dwnOptimModelTempPrecond = calculateDiagPrecondDwnSystem(dwnOptimModelTemp, dwnDualHessianTemp);
    
    [~, detailsDwnHessianPrecondTemp] = calculateDwnDualHessian(dwnOptimModelTempPrecond, treeDataTemp,...
        dwnOptimCost, optionPrecondition);
    detailsPrecond.norm = detailsDwnHessianPrecondTemp.norm;
    diagDualHessianTemp = diag(diag(dwnDualHessianTemp).^(-0.5));
    
    prob = sqrt(treeData.prob);
    dwnOptimModelPrecond = dwnOptimModel;
    
    for j = 1:nStage 
        dwnOptimModelPrecond.G{j,1} = dwnOptimModel.G{j,1};
        dwnOptimModelPrecond.F{j,1} = dwnOptimModel.F{j,1};
        
        k = treeData.stage(j) + 1;
        dwnOptimModelPrecond.G{j,1} = prob(j)*diagDualHessianTemp((k-1)*nz+1:(k-1)*nz+nu, (k-1)*nz+1:(k-1)*nz+nu)*...
            dwnOptimModel.G{j,1};
        dwnOptimModelPrecond.umax((j-1)*nu+1:j*nu, 1) = prob(j)*diagDualHessianTemp((k-1)*nz+1:(k-1)*nz+nu,...
            (k-1)*nz+1:(k-1)*nz+nu)*dwnOptimModel.umax((j-1)*nu+1:j*nu, 1);
        dwnOptimModelPrecond.umin((j-1)*nu+1:j*nu, 1) = prob(j)*diagDualHessianTemp((k-1)*nz+1:(k-1)*nz+nu,...
            (k-1)*nz+1:(k-1)*nz+nu)*dwnOptimModel.umin((j-1)*nu+1:j*nu, 1);
        
        dwnOptimModelPrecond.F{j,1} = prob(j)*diagDualHessianTemp((k-1)*nz+nu+1:k*nz,(k-1)*nz+nu+1:k*nz)*...
            dwnOptimModel.F{j,1};
        dwnOptimModelPrecond.xmax((j-1)*nx+1:j*nx, 1) = prob(j)*diagDualHessianTemp((k-1)*nz+nu+1:(k-1)*nz+nu+nx,...
            (k-1)*nz+nu+1:(k-1)*nz+nu+nx)*dwnOptimModel.xmax((j-1)*nx+1:j*nx, 1);
        dwnOptimModelPrecond.xmin((j-1)*nx+1:j*nx, 1) = prob(j)*diagDualHessianTemp((k-1)*nz+nu+1:(k-1)*nz+nu+nx,...
            (k-1)*nz+nu+1:(k-1)*nz+nu+nx)*dwnOptimModel.xmin((j-1)*nx+1:j*nx, 1);
        dwnOptimModelPrecond.xs((j-1)*nx+1:j*nx, 1) = prob(j)*diagDualHessianTemp((k-1)*nz+nu+nx+1:k*nz,...
            (k-1)*nz+nu+nx+1:k*nz)*dwnOptimModel.xs((j-1)*nx+1:j*nx, 1);
    end
end
detailsPrecond.mat_diag_prcnd = diag(dwnDualHessianTemp).^(-0.5);

end


