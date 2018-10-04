function [ model ] = gurobiDwnUpdate( model, sys, Tree, V, modelOpts)
%  
% Update the gurobi optimisation model based on the current state, nominal
% demand, nominal prices and 
% 
% INPUT     :     model  :   Gurobi optimisation model
%                 sys    :   structure containing the system description 
%                 Tree   :   tree structure that describes the uncertainty 
%                 V      :   cost function including the soft costs
%
% OUTPUT    :     model  :   updated Gurobi optimisation model with the 
%                            nominal demand, nominal forecast, current state 
%                            and previous control.
%

if(isfield(Tree, 'errorPrice'))
    PRICE_FLAG = 1;
else
    PRICE_FLAG = 0;
end

Nd = size(Tree.stage,1);
nu = sys.nu;
nx = sys.nx;
nvar = 2*nx+nu;
ne = size(modelOpts.E,1);
kRowEquality = 2*Nd*nx;

for i = 1:Nd
    if(i == 1)
        model.rhs(1:nx,1) = sys.Gd*modelOpts.demand(1,:)'+modelOpts.x;
        model.rhs(kRowEquality + 1:kRowEquality + ne,1) = -modelOpts.Ed*modelOpts.demand(1,:)';
        model.obj(1:nu) = V.alpha(1,:)' - 2*V.Wu*modelOpts.uprev;
    else
        nodeStage = Tree.stage(i) + 1;
        if(PRICE_FLAG)
            model.obj((i-1)*nvar+1:(i-1)*nvar+nu) = Tree.prob(i, 1)*(V.alpha(nodeStage, :)' + Tree.errorPrice(i, :)');
        else 
            model.obj((i-1)*nvar+1:(i-1)*nvar+nu) = Tree.prob(i, 1)* V.alpha(nodeStage,:)';
        end 
        model.rhs((i-1)*nx+1:i*nx,1) = sys.Gd*(modelOpts.demand(nodeStage,:)' + Tree.value(i,:)');
        model.rhs(kRowEquality+(i-1)*ne+1:kRowEquality+i*ne,1) = ...
            -modelOpts.Ed*(modelOpts.demand(nodeStage,:)'+Tree.value(i,:)');       
    end
end

end







