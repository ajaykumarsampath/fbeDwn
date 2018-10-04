function [ model] = gurobiDwnModel( sys, Tree, V, modelOpts)
%  
% Generate the gurobi for drining water network in the effinet problem 
% with soft constraints
% 
% INPUT     :     sys       :   contains the system description 
%                 Tree      :   tree structure used to describe the uncertainty in
%                               the forcasts
%                 V         :   cost function including the soft costs
%                 modelOpts :   conatin the options for the model
%
% OUTPUT    :     model     :   Gurobi optimisation model 


Nd = size(Tree.stage,1);
K  = size(Tree.leaves,1);
nu = sys.nu;
nx = sys.nx;
nz = nx+nu;
nvar = 2*nx+nu;
ne=size(modelOpts.E,1);
Qs = 10000*eye(nx);

model.Q = sparse(Nd*(nz+nx));
model.obj = zeros(Nd*(nz+nx),1);
model.A = sparse(2*Nd*nx+Nd*ne);
model.rhs = zeros(2*Nd*nx+Nd*ne,1);
model.sense=[repmat('=', Nd*nx, 1); repmat('>', Nd*nx, 1); repmat('=', Nd*ne, 1)];
%model.varnames = cell(1);

Qsps = sparse(Qs);
Rsps = sparse(V.Wu);
kRowDynamics = Nd*nx;
kRowEquality = 2*Nd*nx;

for iNode =1:Nd
    if(iNode == 1)
        probBar = Tree.prob(iNode) + sum(Tree.prob(Tree.children{iNode}));
        model.Q((iNode-1)*nvar+1:(iNode-1)*nvar+nu, (iNode-1)*nvar+1:(iNode-1)*nvar+nu) = probBar*Rsps;
        model.Q((iNode-1)*nvar+nz+1:iNode*nvar, (iNode-1)*nvar+nz+1:iNode*nvar) = Qsps;
        model.lb(1:nvar,1) = [sys.umin(1:nu,1);sys.xmin(1:nx);zeros(nx,1)];
        model.ub(1:nvar,1) = [sys.umax(1:nu,1);sys.xmax(1:nx);sys.xmax(1:nx)];
        model.A(1:nx,1:nz) = sparse([-sys.B eye(nx)]);
        model.A(kRowDynamics+1:kRowDynamics+nx,nu+1:nvar) = sparse([eye(nx) eye(nx)]);
        model.rhs(kRowDynamics+1:kRowDynamics+nx,1) = sys.xs(1:nx);
        model.A(kRowEquality+1:kRowEquality+ne,1:nu) = sparse(modelOpts.E);
    else
        iAncestor = Tree.ancestor(iNode);
        iStage = Tree.stage(iNode) + 1;
        if(iNode <= Nd-K)
            probBar = Tree.prob(iNode) + sum(Tree.prob(Tree.children{iNode}));
            model.Q((iNode-1)*nvar+1:(iNode-1)*nvar+nu, (iNode-1)*nvar+1:(iNode-1)*nvar+nu) = probBar*Rsps;
        else
            probBar = Tree.prob(iNode);
            model.Q((iNode-1)*nvar+1:(iNode-1)*nvar+nu, (iNode-1)*nvar+1:(iNode-1)*nvar+nu) = probBar*Rsps;
        end
        model.Q((iNode-1)*nvar+1:(iNode-1)*nvar+nu, (iAncestor-1)*nvar+1:(iAncestor-1)*nvar+nu)...
            = -Tree.prob(iNode)*Rsps;
        model.Q((iAncestor-1)*nvar+1:(iAncestor-1)*nvar+nu, (iNode-1)*nvar+1:(iNode-1)*nvar+nu)...
            = -Tree.prob(iNode)*Rsps;
        
        model.Q((iNode-1)*nvar+nz+1:iNode*nvar,(iNode-1)*nvar+nz+1:iNode*nvar) = Qsps;
        %model.lb((i-1)*nvar+1:i*nvar,1) = [sys.umin((i-1)*nu+1:i*nu);sys.xmin((i-1)*nx+1:i*nx);zeros(nx,1)];
        %model.ub((i-1)*nvar+1:i*nvar,1) = [sys.umax((i-1)*nu+1:i*nu);sys.xmax((i-1)*nx+1:i*nx);...
         %   sys.xmax((i-1)*nx+1:i*nx)];
        model.lb((iNode-1)*nvar+1:iNode*nvar,1) = [sys.umin((iStage - 1)*nu+1:iStage*nu);...
            sys.xmin((iStage-1)*nx+1:iStage*nx);zeros(nx,1)];
        model.ub((iNode-1)*nvar+1:iNode*nvar,1) = [sys.umax((iStage - 1)*nu+1:iStage*nu);...
            sys.xmax((iStage-1)*nx+1:iStage*nx);sys.xmax((iStage-1)*nx+1:iStage*nx)];
        model.A((iNode-1)*nx+1:iNode*nx,(iNode-1)*nvar+1:(iNode-1)*nvar+nz) = sparse([-sys.B eye(nx)]);
        model.A((iNode-1)*nx+1:iNode*nx,iAncestor*nvar-2*nx+1:iAncestor*nvar-nx) = -speye(nx);
        model.A(kRowDynamics+(iNode-1)*nx+1:kRowDynamics+iNode*nx,...
            (iNode-1)*nvar+nu+1:iNode*nvar) = sparse([eye(nx) eye(nx)]);
        model.rhs(kRowDynamics+(iNode-1)*nx+1:kRowDynamics+iNode*nx,1)...
            = sys.xs((iStage-1)*nx+1:iStage*nx,1);
        model.A(kRowEquality+(iNode-1)*ne+1:kRowEquality+iNode*ne,(iNode-1)*nvar+1:(iNode-1)*nvar+nu) = sparse(modelOpts.E);
    end
end

end





