function [ Z, detailsApg] = solveSmpcDwnWithApg(dwnOptimModel, dwnFactorStepModel,...
    treeData, dwnOptimCost, optionDwnApg) 
% APG_effinet_dist_box 
% This function is implements the dual proximal gradient algorithm to solve 
% the sc-mpc of the DWN with soft constraints on state and hard constraits
% on input.
%
% Syntax 
%  
% INPUT   
%   dwnOptimalModel      :  dwn model for the smpc optimisation 
%   dwnFactorStepModel   :  offline matrices for factor step
%   treeData             :  uncertainty data 
%   dwnOptimCost         :  cost function
%   optionDwnApg         :  options for the dual APG algorithm
%             Termination options: maximum number of steps etc   
%             Type of constraints: soft or hard constaints
%             Intial condition and preious control input.
% 

nScenario = length(treeData.leaves);% total scenarios in the tree
nNode = length(treeData.stage);%toal nodes in the tree

% Initalizing the dual varibables
ny_x = size(dwnOptimModel.F{1}, 1);
ny_u = size(dwnOptimModel.G{1}, 1);

theta = [1 1]';
Y.x0 = zeros(ny_x, nNode);
Y.u0 = zeros(ny_u, nNode);
Y.x1 = zeros(ny_x, nNode);
Y.u1 = zeros(ny_u, nNode);
W.x = zeros(ny_x, nNode);
W.u = zeros(ny_u, nNode);

detailsApg.term_crit = zeros(1,4);

optionProximal.lambda = optionDwnApg.lambda;
optionProximal.xmin = reshape(dwnOptimModel.xmin, dwnOptimModel.nx, nNode);
optionProximal.xs = reshape(dwnOptimModel.xs, dwnOptimModel.nx, nNode);
optionProximal.xmax = reshape(dwnOptimModel.xmax, dwnOptimModel.nx, nNode);

optionProximal.umin = reshape(dwnOptimModel.umin, dwnOptimModel.nu, nNode);
optionProximal.umax = reshape(dwnOptimModel.umax, dwnOptimModel.nu, nNode);
optionProximal.nx = dwnOptimModel.nx;
optionProximal.nu = dwnOptimModel.nu;
optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnApg.lambda;
optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnApg.lambda;
optionProximal.iter = 200;
optionProximal.constraints = optionDwnApg.constraints;

uprev = dwnOptimModel.L*optionDwnApg.state.v + optionDwnApg.state.prev_vhat;

prm_feas.x = zeros(ny_x, nNode);
prm_feas.u = zeros(ny_u, nNode);

tic
iStep = 1;
jobj = zeros(1, optionDwnApg.steps-1);
while(iStep < optionDwnApg.steps)
    % Step 1: accelerated step
    W.x = Y.x1 + theta(2)*(1/theta(1)-1)*(Y.x1 - Y.x0);
    W.u = Y.u1 + theta(2)*(1/theta(1)-1)*(Y.u1 - Y.u0);
    
    % step 2: calculating the dual gradient
    [Z, ~] = calculateDwnSolveStep(dwnOptimModel, treeData, dwnFactorStepModel, W,...
        optionDwnApg.x, optionDwnApg.state);
    
    %
    for i = 1:nNode - nScenario + 1
        if(i == 1)
            jobj(iStep) = jobj(iStep) + dwnOptimCost.alpha(1,:)*Z.U(:,1) + (Z.U(:,1)-uprev)'*...
                dwnOptimCost.Wu*(Z.U(:,1) - uprev) + W.x(:,1)'*dwnOptimModel.F{i}*Z.X(:,2) +...
                W.u(:,1)'*dwnOptimModel.G{1}*Z.U(:,1);
        else
            stage = treeData.stage(i-1) + 1;
            nchild = treeData.children{i-1};
            for l = 1:length(nchild)
                deltaU = Z.U(:, nchild(l)) - Z.U(:, treeData.ancestor(nchild(l)));
                jobj(iStep) = jobj(iStep) + treeData.prob(nchild(l), 1)*( dwnOptimCost.alpha(stage + 1, :)*...
                    Z.U(:,nchild(l)) + deltaU'*dwnOptimCost.Wu*deltaU ) +...
                    W.x(:, nchild(l))'*dwnOptimModel.F{nchild(l)}*Z.X(:, nchild(l) + 1)+...
                    W.u(:, nchild(l))'*dwnOptimModel.G{nchild(l)}*Z.U(:, nchild(l));
            end
        end
    end
    
    % step 3: proximal with respect to g
    t = calculateDwnProximalStep(Z, W, dwnOptimModel, treeData, optionProximal);
    
    % step 4: update the dual vector
    Y.x0 = Y.x1;
    Y.u0 = Y.u1;
    
    if(dwnOptimModel.cell)
        for i = 1:nNode
            Y.x1(:, i) = W.x(:,i) + optionDwnApg.lambda*(dwnOptimModel.F{i,1}*...
                Z.X(:, i+1) - t.x(:,i));
            Y.u1(:, i) = W.u(:,i) + optionDwnApg.lambda*(dwnOptimModel.G{i,1}*...
                Z.U(:, i) - t.u(:,i));
            prm_feas.x(:, i) = dwnOptimModel.F{i,1}*Z.X(:, i + 1);
            prm_feas.u(:, i) = dwnOptimModel.G{i,1}*Z.U(:, i);
        end
    else
        Y.x1 = W.x + optionDwnApg.lambda*(dwnOptimModel.F*Z.X(:, 2:nNode) - t.x);
        Y.u1 = W.u + optionDwnApg.lambda*(dwnOptimModel.G(dwnOptimModel.nx + 1:end, :)*Z.U - t.u);
    end
    
    iter = iStep;
    detailsApg.prm_cst(iter) = 0;%primal cost;
    detailsApg.dual_cst(iter) = 0;% dual cost;
    
    theta(1) = theta(2);
    theta(2) = (sqrt(theta(1)^4 + 4*theta(1)^2) - theta(1)^2)/2;
    iStep = iStep + 1;
end
%figure
%plot(jobj);
detailsApg.gpad_solve = toc;
detailsApg.W = W;
detailsApg.jobj = jobj;
detailsApg.prm_feas = prm_feas;
detailsApg.t = t;
detailsApg.Y = Y;

end







