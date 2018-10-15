function [ dwnOptimModelPrecond ] = calculateDiagPrecondDwnSystem( dwnOptimModel, dualHessian, optionDiagPrecond)
%
% calculateDiagPrecondDwnSystem calcuate the diagonal preconditioning of the system with 
%  single scenario tree. 
%
% INPUT 
%   dwnOptimModel     : 
%   dualHessian       : 
%   optionDiagPrecond :
%
% OUTPUT 
%   dwnOptimModelPrecond  :
%

if(nargin < 3)
    optionDiagPrecond.dual_hessian = 1;
end 

nx = size(dwnOptimModel.A, 1);
nu = size(dwnOptimModel.B, 2);
Np = dwnOptimModel.Np;

if(optionDiagPrecond.dual_hessian)
    diagDualHessian = diag(diag(dualHessian).^(-0.5));
else
    diagDualHessian = dualHessian;
end

dwnOptimModelPrecond = dwnOptimModel;
if(dwnOptimModel.cell)
    %nz = 2*nx + nu;
    nz = size(dwnOptimModel.F{1}, 1) + size(dwnOptimModel.G{1}, 1);
    rowMatG = size(dwnOptimModel.G{1}, 1);
    %rowMatEd = size(dwnOptimModel.G{1}, 1) - nu;
    for j = 1:Np
        dwnOptimModelPrecond.F{j,1} = dwnOptimModel.F{j,1};
        dwnOptimModelPrecond.G{j,1} = dwnOptimModel.G{j,1};
        dwnOptimModelPrecond.G{j,1} = diagDualHessian((j-1)*nz + 1:(j-1)*nz + rowMatG,...
            (j-1)*nz + 1:(j-1)*nz + nu)*dwnOptimModelPrecond.G{j,1};
        dwnOptimModelPrecond.F{j,1} = diagDualHessian((j-1)*nz + rowMatG + 1:j*nz,...
            (j-1)*nz+nu+1:j*nz)*dwnOptimModelPrecond.F{j,1};
        
        dwnOptimModelPrecond.umax((j-1)*nu + 1:j*nu, 1) = diagDualHessian((j-1)*nz + 1:(j-1)*nz + nu,...
            (j-1)*nz + 1:(j-1)*nz + nu)*dwnOptimModelPrecond.umax((j-1)*nu + 1:j*nu, 1);
        dwnOptimModelPrecond.umin((j-1)*nu + 1:j*nu, 1) = diagDualHessian((j-1)*nz + 1:(j-1)*nz + nu,...
            (j-1)*nz + 1:(j-1)*nz + nu)*dwnOptimModelPrecond.umin((j-1)*nu + 1:j*nu, 1);
        
        dwnOptimModelPrecond.xmax((j-1)*nx + 1:j*nx, 1) = diagDualHessian((j-1)*nz + rowMatG + 1:(j-1)*nz + nu...
            + nx, (j-1)*nz + nu + 1:(j-1)*nz + nu + nx)*dwnOptimModelPrecond.xmax((j-1)*nx + 1:j*nx);
        dwnOptimModelPrecond.xmin((j-1)*nx + 1:j*nx, 1) = diagDualHessian((j-1)*nz + nu + 1:(j-1)*nz + nu...
            +nx, (j-1)*nz + nu + 1:(j-1)*nz + nu + nx)*dwnOptimModelPrecond.xmin((j-1)*nx + 1:j*nx);
        dwnOptimModelPrecond.xs((j-1)*nx + 1:j*nx, 1) = diagDualHessian((j-1)*nz + nu + nx + 1:j*nz,...
            (j-1)*nz + nu + nx + 1:j*nz)*dwnOptimModelPrecond.xmin((j-1)*nx + 1:j*nx);
    end
else
    dwnOptimModelPrecond.F = cell(Np,1);
    dwnOptimModelPrecond.G = cell(Np,1);
    for j = 1:Np
        dwnOptimModelPrecond.F{j,1} = dwnOptimModel.F;
        dwnOptimModelPrecond.G{j,1} = dwnOptimModel.G;
        dwnOptimModelPrecond.G{j,1}(nx + 1:nz,:) = diagDualHessian((j-1)*nz + 1:(j-1)*nz+nu,...
            (j-1)*nz + 1:(j-1)*nz + nu)*dwnOptimModelPrecond.G{j,1};
        dwnOptimModelPrecond.F{j,1}(1:nx, :) = diagDualHessian((j-1)*nz + nu + 1:j*nz,...
            (j-1)*nz + nu + 1:j*nz)*dwnOptimModelPrecond.F{j,1};
        
        dwnOptimModelPrecond.umax((j-1)*nu + 1:j*nu, 1) = diagDualHessian((j-1)*nz + 1:(j-1)*nz + nu,...
            (j-1)*nz + 1:(j-1)*nz + nu)*dwnOptimModelPrecond.umax((j-1)*nu + 1:j*nu, 1);
        dwnOptimModelPrecond.umin((j-1)*nu+1:j*nu, 1) = diagDualHessian((j-1)*nz + 1:(j-1)*nz + nu,...
            (j-1)*nz + 1:(j-1)*nz + nu)*dwnOptimModelPrecond.umin((j-1)*nu + 1:j*nu, 1);
        
        dwnOptimModelPrecond.xmax((j-1)*nx + 1:j*nx, 1)=diagDualHessian((j-1)*nz + nu + 1:(j-1)*nz + nu + nx,...
            (j-1)*nz + nu + 1:(j-1)*nz + nu + nx)*dwnOptimModelPrecond.xmax((j-1)*nx + 1:j*nx);
        dwnOptimModelPrecond.xmin((j-1)*nx+1:j*nx, 1) = diagDualHessian((j-1)*nz + nu + 1:(j-1)*nz + nu + nx,...
            (j-1)*nz + nu + 1:(j-1)*nz + nu + nx)*dwnOptimModelPrecond.xmin((j-1)*nx + 1:j*nx);
        dwnOptimModelPrecond.xs((j-1)*nx + 1:j*nx, 1) = diagDualHessian((j-1)*nz + nu + nx + 1:j*nz,...
            (j-1)*nz + nu + nx + 1:j*nz)*dwnOptimModelPrecond.xmin((j-1)*nx + 1:j*nx);
    end
end

end






