function [ dualHessian, details] = calculateDwnDualHessianVersion(dwnOptimModel, treeData, ...
    dwnOptimCost, optionApgPrecond)
% This function calculates the dual hessian for the DWN model used for the
% dual proximal gradient based algorithm
%
% INPUT 
%   dwnOptimModel    :
%   treeData         :
%   dwnOptimCost     :
%   optionApgPrecond :
%
% OUTPUT 
%   Dual_hessian :
%   details      :
%
%

nx = dwnOptimModel.nx;
nu = dwnOptimModel.nu;
ne = size(optionApgPrecond.E, 1);
Np = dwnOptimModel.Np;
nz = nx + nu;

Bbar = zeros(Np*(nx+ne), Np*nz);
H = zeros(nz*Np);
epsilon = 0e-4*eye(nz);


for i = 1:Np
    if(i == 1)
        Bbar(1:nx,1:nz) = [dwnOptimModel.B -eye(nx)];
        Bbar(Np*nx + (i-1)*ne + 1:Np*nx + i*ne, 1:nu) = optionApgPrecond.E;
    else
        Bbar((i-1)*nx + 1:i*nx, nu + (i-2)*nz + 1:i*nz) = ...
            [dwnOptimModel.A dwnOptimModel.B -eye(nx)];
        Bbar(Np*nx + (i-1)*ne + 1:Np*nx + i*ne, (i-1)*nz + 1:(i-1)*nz + nu) = ...
            optionApgPrecond.E;
    end 
end

for i = 1:Np
    if(i == Np)
        H((i-1)*nz + 1:(i-1)*nz + nu, (i-1)*nz + 1:(i-1)*nz + nu) = ...
            treeData.prob(i)*dwnOptimCost.Wu;
        H((i-1)*nz + 1:(i-1)*nz + nu, (i-2)*nz + 1:(i-2)*nz + nu) = ...
            -treeData.prob(i)*dwnOptimCost.Wu;
        H((i-1)*nz + nu + 1:i*nz, (i-1)*nz + nu + 1:i*nz) = epsilon(nu + 1:nz,...
            nu + 1:nz);
    else
        if(i > 1)
            H((i-1)*nz + 1:(i-1)*nz + nu, (i-2)*nz + 1:(i-2)*nz + nu) = -treeData.prob(i)*...
                dwnOptimCost.Wu;
        end
        H((i-1)*nz + 1:(i-1)*nz + nu, (i-1)*nz + 1:(i-1)*nz + nu) = (treeData.prob(i)...
            + treeData.prob(i+1))*dwnOptimCost.Wu;
        H((i-1)*nz + 1:(i-1)*nz + nu, i*nz + 1:i*nz + nu) = -treeData.prob(i)*dwnOptimCost.Wu;
        H((i-1)*nz + nu + 1:i*nz, (i-1)*nz + nu + 1:i*nz) = epsilon(nu+1:nz,nu+1:nz);
    end
end

L = null(Bbar);
Wbar = L'*H*L;

if(dwnOptimModel.cell)
    nz1 = size(dwnOptimModel.F{1}, 1) + size(dwnOptimModel.G{1}, 1);
    A = zeros(Np*nz1, Np*nz);
    for i = 1:Np
        A((i-1)*nz1 + 1:i*nz1, (i-1)*nz + 1:i*nz) = blkdiag(dwnOptimModel.G{i}, dwnOptimModel.F{i});
    end
else
    nz1 = size(dwnOptimModel.F, 1) + size(dwnOptimModel.G, 1);
    for i = 1:Np
        A((i-1)*nz1 + 1:i*nz1, (i-1)*nz + 1:i*nz) = blkdiag(dwnOptimModel.G, dwnOptimModel.F);
    end
end

Abar = A*L;
K1 = Wbar\(eye(size(Wbar,1)));
details.cond_primal = cond(K1);
details.min_eig = min(eig(K1));

dualHessian = Abar*K1*Abar';
details.condition_number = cond(dualHessian);
details.norm = norm(dualHessian);

end




