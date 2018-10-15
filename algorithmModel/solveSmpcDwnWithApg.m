function [  dwnSmpcZvar, detailsAlgoApg] = solveSmpcDwnWithApg(dwnOptimModel, dwnFactorStepModel,...
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

%{
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

uPrev = dwnOptimModel.L*optionDwnApg.state.v + optionDwnApg.state.prev_vhat;

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
    [Z, ~] = calculateDwnSolveStep(dwnOptimModel, treeData, W, dwnFactorStepModel,...
        optionDwnApg.x, optionDwnApg.state);
    
    for i = 1:nNode - nScenario + 1
        if(i == 1)
            jobj(iStep) = jobj(iStep) + dwnOptimCost.alpha(1,:)*Z.U(:,1) + (Z.U(:,1)-uPrev)'*...
                dwnOptimCost.Wu*(Z.U(:,1) - uPrev) + W.x(:,1)'*dwnOptimModel.F{i}*Z.X(:,2) +...
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
    [t, proximalCost] = calculateDwnProximalStep(dwnOptimModel, treeData, W, Z, optionProximal);
    
    dwnSmpcApgResidual = calculateDwnFbeResidual(dwnOptimModel, Z, t);
    optionDwnApgObjective.previousU = uPrev;
    optionDwnApgObjective.lambda = optionDwnApg.lambda;
    optionDwnApgObjective.dualVar = W;
    optionDwnApgObjective.dwnSmpcZvar = Z;
    optionDwnApgObjective.proximalCost = proximalCost;
    optionDwnApgObjective.fbeResidual = dwnSmpcApgResidual;
    apgObjDualY = calculateDwnDualEnvelopObjective( dwnOptimCost, treeData,...
        optionDwnApgObjective);
    detailsApg.apgObjDualY(iStep) = apgObjDualY;
    
    residual = [dwnSmpcApgResidual.x; dwnSmpcApgResidual.u];
    dimResidual = size(residual);
    dualVarVec = reshape([W.x;W.u], dimResidual(1)*dimResidual(2), 1);
    residualVec = reshape(residual, dimResidual(1)*dimResidual(2), 1);
    detailsApg.penaltyCost(iStep) = -dualVarVec'*residualVec;
    detailsApg.normSmpcResidual(iStep) = norm(residualVec);
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
    if(mod(iStep, 100) == 0)
        optionDwnApg.lambda = optionDwnApg.lambda/1.3;
        optionProximal.lambda = optionDwnApg.lambda;
        optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnApg.lambda;
        optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnApg.lambda;
    end
    iStep = iStep + 1;
end
detailsApg.gpad_solve = toc;
detailsApg.W = W;
detailsApg.jobj = jobj;
detailsApg.prm_feas = prm_feas;
detailsApg.t = t;
detailsApg.Y = Y;
%}

nNodes = length(treeData.stage);%toal nodes in the tree
nDualx = size(dwnOptimModel.F{1}, 1);
nDualu = size(dwnOptimModel.G{1}, 1);

optionProximal.lambda = optionDwnApg.lambda;
optionProximal.xmin = reshape(dwnOptimModel.xmin, dwnOptimModel.nx, nNodes);
optionProximal.xs = reshape(dwnOptimModel.xs, dwnOptimModel.nx, nNodes);
optionProximal.xmax = reshape(dwnOptimModel.xmax, dwnOptimModel.nx, nNodes);
optionProximal.umin = reshape(dwnOptimModel.umin, dwnOptimModel.nu, nNodes);
optionProximal.umax = reshape(dwnOptimModel.umax, dwnOptimModel.nu, nNodes);
optionProximal.nx = dwnOptimModel.nx;
optionProximal.nu = dwnOptimModel.nu;
optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnApg.lambda;
optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnApg.lambda;
optionProximal.iter = 200;
optionProximal.constraints = optionDwnApg.constraints;

detailsAlgoApg.terminationCriteria = zeros(1,4);

dualVarPrevY.x = zeros(nDualx, nNodes);
dualVarPrevY.u = zeros(nDualu, nNodes);

dualVarY.x = zeros(nDualx, nNodes);
dualVarY.u = zeros(nDualu, nNodes);

dualVarW.x = zeros(nDualx, nNodes);
dualVarW.u = zeros(nDualu, nNodes);

primalFeasible.x = zeros(nDualx, nNodes);
primalFeasible.u = zeros(nDualu, nNodes);

uPrev = dwnOptimModel.L*optionDwnApg.state.v + optionDwnApg.state.prev_vhat;
theta = [1 1]'; 

tic
iStep = 1;
while( iStep < optionDwnApg.steps )
    % step 1: accelerating dual vector
    dualVarW.x = dualVarY.x + theta(2)*(1/theta(1) - 1)*(dualVarY.x - dualVarPrevY.x);
    dualVarW.u = dualVarY.u + theta(2)*(1/theta(1) - 1)*(dualVarY.u - dualVarPrevY.u);
    % step 2: calculate the gradient of the conjugate f
    dwnSmpcZvar = calculateDwnSolveStep(dwnOptimModel, treeData, dualVarW, dwnFactorStepModel,...
        optionDwnApg.x, optionDwnApg.state);
    % step 3 : calculate the proximal of g
    [dwnSmpcTvar, proximalCost] = calculateDwnProximalStep(dwnOptimModel, treeData, dualVarW,...
        dwnSmpcZvar, optionProximal);
    % step 4 : calculate the residual of global Fbe and gradient of the dual Fbe
    dwnSmpcResidual = calculateDwnFbeResidual(dwnOptimModel, dwnSmpcZvar, dwnSmpcTvar);
    
    optionDwnApgObjective.previousU = uPrev;
    optionDwnApgObjective.lambda = optionDwnApg.lambda;
    optionDwnApgObjective.dualVar = dualVarW;
    optionDwnApgObjective.dwnSmpcZvar = dwnSmpcZvar;
    optionDwnApgObjective.proximalCost = proximalCost;
    optionDwnApgObjective.fbeResidual = dwnSmpcResidual;
    residual = [dwnSmpcResidual.x; dwnSmpcResidual.u];
    dimResidual = size(residual);
    
    fbeObjDualY = calculateDwnDualEnvelopObjective( dwnOptimCost, treeData, optionDwnApgObjective);
    detailsAlgoApg.fbeObjDualY(iStep) = fbeObjDualY;
    detailsAlgoApg.normSmpcResidual(iStep) = norm(reshape(residual, dimResidual(1)*dimResidual(2), 1));
    
    dualVarPrevY = dualVarY;
    dualVarY.x = dualVarW.x - optionDwnApg.lambda*dwnSmpcResidual.x;
    dualVarY.u = dualVarW.u - optionDwnApg.lambda*dwnSmpcResidual.u;
    
    iStep = iStep + 1;
    theta(1) = theta(2);
    theta(2) = (sqrt(theta(1)^4 + 4*theta(1)^2) - theta(1)^2)/2; 
    
end

detailsAlgoApg.gpad_solve = toc;
detailsAlgoApg.dualVarW = dualVarW;
detailsAlgoApg.dualVar = dualVarY;
detailsAlgoApg.dwnSmpcTvar = dwnSmpcTvar;

end







