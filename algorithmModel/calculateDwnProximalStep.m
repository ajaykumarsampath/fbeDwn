function [dwnSmpcTvar, proximalCost] = calculateDwnProximalStep(dwnOptimModel, treeData,...
    dualW, dwnSmpcZvar, optionProximal)
%
% This function calcualte the proximal for the state and input constraints
% soft constraits on state, 
% hard constraints on input.
%
% INPUT-----   Z   :  state and control
%              W   :  dual variable
%      opts_prox   :  proximal options

distance = [0;0];
if(dwnOptimModel.cell)
    constraints.x = dualW.x/optionProximal.lambda;
    constraints.u = dualW.u/optionProximal.lambda;
    
    nNode = size(treeData.stage,1);
    for i = 1:nNode
        constraints.x(:,i) = constraints.x(:,i) + dwnOptimModel.F{i,1}*dwnSmpcZvar.X(:,i+1);
        constraints.u(:,i) = constraints.u(:,i) + dwnOptimModel.G{i,1}*dwnSmpcZvar.U(:,i);
    end
    
    % hard constraints on input u
    dwnSmpcTvar.u = zeros(size(dualW.u, 1), size(constraints.u, 2));
    dwnSmpcTvar.u(1:dwnOptimModel.nu,:) = min(optionProximal.umax,...
        constraints.u(1:dwnOptimModel.nu,:));
    dwnSmpcTvar.u(1:dwnOptimModel.nu,:) = max(optionProximal.umin,...
        dwnSmpcTvar.u(1:dwnOptimModel.nu,:));
    rowEd = size(constraints.u, 1) - dwnOptimModel.nu;
    if(rowEd)
        dwnSmpcTvar.u(dwnOptimModel.nu + 1:end,:) = reshape(dwnOptimModel.equalEd, rowEd, nNode);
    end
    
    % soft constraints on input x
    optionProximal.xmax = reshape(optionProximal.xmax, dwnOptimModel.nx*size(constraints.x,2), 1);
    optionProximal.xmin = reshape(optionProximal.xmin, dwnOptimModel.nx*size(constraints.x,2), 1);
    optionProximal.xs = reshape(optionProximal.xs, dwnOptimModel.nx*size(constraints.x,2), 1);
    
    dwnSmpcTvar.x = zeros(2*dwnOptimModel.nx, size(constraints.x, 2));
    xtemp = reshape(constraints.x(1:dwnOptimModel.nx,:), dwnOptimModel.nx*nNode, 1);
    
    projXset = min(optionProximal.xmax, xtemp);
    projXset = max(optionProximal.xmin, projXset);
    
    distance(1) = norm(xtemp - projXset, 2);
    if(distance(1) > optionProximal.gamma_xbox)
        disp('proximal distance')
        xtemp = xtemp + optionProximal.gamma_xbox*(projXset - xtemp)/distance(1);
    else
        xtemp = projXset;
    end
    
    dwnSmpcTvar.x(1:dwnOptimModel.nx,:) = reshape(xtemp, dwnOptimModel.nx, nNode);
    projXset = min(optionProximal.xmax, xtemp);
    projXset = max(optionProximal.xmin, projXset);
    proximalCost.distanceXset = optionProximal.lambda*optionProximal.gamma_xbox*...
        norm(xtemp - projXset, 2);
    
    xtemp = reshape(constraints.x(dwnOptimModel.nx+1:2*dwnOptimModel.nx,:), dwnOptimModel.nx*nNode, 1);
    projXsafeSet = max(optionProximal.xs, xtemp); 
    distance(2) = norm(xtemp-projXsafeSet, 2);
    if(distance(2) > optionProximal.gamma_xs)
        disp('proximal distance')
        xtemp = xtemp + optionProximal.gamma_xs*(projXsafeSet - xtemp)/distance(2);
    else
        xtemp = projXsafeSet;
    end
    
    dwnSmpcTvar.x(dwnOptimModel.nx + 1:2*dwnOptimModel.nx, :) = reshape(xtemp, dwnOptimModel.nx, nNode);
    projXsafeSet = max(optionProximal.xs, xtemp);
    proximalCost.distanceSafe = optionProximal.lambda*optionProximal.gamma_xs*...
        norm(xtemp - projXsafeSet, 2);
else
    constraints.x = [dualW.y(1:optionProximal.nx,2:end) dualW.yt]/optionProximal.lambda +...
        dwnOptimModel.F(1:dwnOptimModel.nx,:)*dwnSmpcZvar.X(:, 2:end);
    constraints.u = dualW.y(optionProximal.nx+1:end,:)/optionProximal.lambda +...
        dwnOptimModel.G(dwnOptimModel.nx+1:end,:)*dwnSmpcZvar.U;
    
    optionProximal.xmax = reshape(optionProximal.xmax, dwnOptimModel.nx, size(constraints.x,2));
    optionProximal.xmin = reshape(optionProximal.xmin, dwnOptimModel.nx, size(constraints.x,2));
    % hard constraints on input u
    dwnSmpcTvar.u = zeros(dwnOptimModel.nu, size(constraints.u,2));
    dwnSmpcTvar.u = min(optionProximal.umax, constraints.u(1:dwnOptimModel.nu,:));
    dwnSmpcTvar.u = max(optionProximal.umin, dwnSmpcTvar.u);
    % hard constraints on state x
    dwnSmpcTvar.x = zeros(dwnOptimModel.nx,size(constraints.x,2));
    dwnSmpcTvar.x = min(optionProximal.xmax, constraints.x(1:dwnOptimModel.nx,:));
    dwnSmpcTvar.x = max(optionProximal.xmin, dwnSmpcTvar.x);
end

end

