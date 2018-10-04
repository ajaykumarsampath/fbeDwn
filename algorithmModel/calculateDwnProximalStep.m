function [ proximalT ] = calculateDwnProximalStep(Z, dualW, dwnOptimModel, treeData, optionProximal)
%
% This function calcualte the proximal for the state and input constraints
% soft constraits on state, 
% hard constraints on input.
%
% INPUT-----   Z   :  state and control
%              W   :  dual variable
%      opts_prox   :  proximal options.
%

distance = [0;0];
if(dwnOptimModel.cell)
    constraints.x = dualW.x/optionProximal.lambda;
    constraints.u = dualW.u/optionProximal.lambda;
    
    nNode = size(treeData.stage,1);
    for i = 1:nNode
        constraints.x(:,i) = constraints.x(:,i) + dwnOptimModel.F{i,1}*Z.X(:,i+1);
        constraints.u(:,i) = constraints.u(:,i) + dwnOptimModel.G{i,1}*Z.U(:,i);
    end
    
    % hard constraints on input u
    proximalT.u = zeros(dwnOptimModel.nu, size(constraints.u, 2));
    proximalT.u = min(optionProximal.umax, constraints.u(1:dwnOptimModel.nu,:));
    proximalT.u = max(optionProximal.umin, proximalT.u);
    
    % soft constraints on input x
    optionProximal.xmax = reshape(optionProximal.xmax, dwnOptimModel.nx*size(constraints.x,2), 1);
    optionProximal.xmin = reshape(optionProximal.xmin, dwnOptimModel.nx*size(constraints.x,2), 1);
    optionProximal.xs = reshape(optionProximal.xs, dwnOptimModel.nx*size(constraints.x,2), 1);
    
    proximalT.x = zeros(2*dwnOptimModel.nx, size(constraints.x, 2));
    xtemp = reshape(constraints.x(1:dwnOptimModel.nx,:), dwnOptimModel.nx*nNode, 1);
    
    projXset = min(optionProximal.xmax, xtemp);
    projXset = max(optionProximal.xmin, projXset);
    
    distance(1) = norm(xtemp-projXset, 2);
    if(distance(1) > optionProximal.gamma_xbox)
        disp('proximal distance')
        xtemp = xtemp + optionProximal.gamma_xbox*(projXset - xtemp)/distance(1);
    else
        xtemp = projXset;
    end
    
    proximalT.x(1:dwnOptimModel.nx,:) = reshape(xtemp, dwnOptimModel.nx, nNode);
    xtemp = reshape(constraints.x(dwnOptimModel.nx+1:2*dwnOptimModel.nx,:), dwnOptimModel.nx*nNode, 1);
    projXsafeSet = max(optionProximal.xs, xtemp); 
    distance(2) = norm(xtemp-projXsafeSet, 2);
    if(distance(2) > optionProximal.gamma_xs)
        disp('proximal distance')
        xtemp = xtemp + optionProximal.gamma_xs*(projXsafeSet - xtemp)/distance(2);
    else
        xtemp = projXsafeSet;
    end
    
    proximalT.x(dwnOptimModel.nx + 1:2*dwnOptimModel.nx, :) = reshape(xtemp, dwnOptimModel.nx, nNode);
else
    constraints.x = [dualW.y(1:optionProximal.nx,2:end) dualW.yt]/optionProximal.lambda +...
        dwnOptimModel.F(1:dwnOptimModel.nx,:)*Z.X(:, 2:end);
    constraints.u = dualW.y(optionProximal.nx+1:end,:)/optionProximal.lambda +...
        dwnOptimModel.G(dwnOptimModel.nx+1:end,:)*Z.U;
    
    optionProximal.xmax = reshape(optionProximal.xmax, dwnOptimModel.nx, size(constraints.x,2));
    optionProximal.xmin = reshape(optionProximal.xmin, dwnOptimModel.nx, size(constraints.x,2));
    % hard constraints on input u
    proximalT.u = zeros(dwnOptimModel.nu, size(constraints.u,2));
    proximalT.u = min(optionProximal.umax, constraints.u(1:dwnOptimModel.nu,:));
    proximalT.u = max(optionProximal.umin, proximalT.u);
    % hard constraints on state x
    proximalT.x = zeros(dwnOptimModel.nx,size(constraints.x,2));
    proximalT.x = min(optionProximal.xmax, constraints.x(1:dwnOptimModel.nx,:));
    proximalT.x = max(optionProximal.xmin, proximalT.x);
end

end

