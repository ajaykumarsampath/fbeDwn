close all;
clear;
clc

kk = 4875;
load('dwn');
%load('dwn_big');
P.Hp = 24;
P.Hu = 23;
Dd = DemandData(:,1); % My data

% Define data
optionTree.Tmax = 800; % time steps for which data are available
optionTree.nScen = 750; % no. of scenarios
optionTree.N = 23; % prediction

load('svm_error.mat');
SVM_ERROR = svm_demand_error(1:optionTree.nScen, 1:optionTree.N);
SVM_ERROR = reshape(SVM_ERROR,optionTree.nScen, 1, optionTree.N);
demandForecastErrorW = SVM_ERROR(1:optionTree.nScen, :, 1:optionTree.N);
optionTree.nw = size(demandForecastErrorW, 2); % no. of component of the stochastic variable

solver.gurobi = 1;
solver.APG = 1;
transferDataCuda = 1;

DemandRatio = zeros(S.nd,1);
for i = 1:S.nd
    DemandRatio(i,1) = DemandData(1,i)./Dd(1,1);
end
%% scenario tree generation
optionTree.ni = ones(optionTree.N,1); % max number of desired scenarios for each time stage
branchingFactor =  [3 2 2];
optionTree.ni(1:length(branchingFactor)) = branchingFactor;
optionTree.Wscaling = 1; % 1/0: do/don't scale disturbance W
optionTree.Wfiltering = 0; % 1/0 do/don't filter disturbance W by system dynamics and u=Kx

[sampleTreeUncertain, detailScenarioTree] = treeFormation(demandForecastErrorW, optionTree);

sampleTreeUncertain{1}.value = kron(DemandRatio', sampleTreeUncertain{1}.value);
sampleTreeUncertain = sampleTreeUncertain{1};
[uncertainTree, new_tree_details] = tranformTree(sampleTreeUncertain);

nScenarios = length(uncertainTree.leaves);
nNode = length(uncertainTree.stage) + 1;
%% 
% transform the effiniet into proximal algorithm 
optionDwnSys.gamma_xs = 1e24;
optionDwnSys.gamma_xbox = 1e29;
optionDwnSys.normalise = 0;
optionDwnSys.cell = 0;
dwnModel = transformDwnOptimisationModel(S, P, uncertainTree, optionDwnSys);
optionDwnSys.cell = 1;

[dwnApgOptimModel, dwnApgOptimCost] = transformDwnOptimisationModel(S, P, uncertainTree, optionDwnSys);
%% Particular solution calculation
dwnCurrentState.demand = 3600*DemandData(kk:kk+P.Hu,:);
prev_vhat = 3600*dwnApgOptimModel.L1*DemandData(kk-1,:)';

dwnCurrentState.prev_vhat = prev_vhat;
dwnApgOptimCost.alpha = (kron(ones(P.Hp,1),P.alpha1')+P.alpha2(kk:kk+P.Hu,:));
dwnApgOptimCost.Qe = 100*eye(dwnModel.nx);
dwnApgOptimCost.Qs = 100*eye(dwnModel.nx);
eliminateCurrentState = calculateParticularSolution(dwnApgOptimModel, uncertainTree, dwnApgOptimCost, dwnCurrentState);
%current_state_opt.v=3600*[0.0656 0.00 0.0849 0.0934]';
eliminateCurrentState.v = 3600*rand(size(dwnModel.L,2), 1);

%% Preconditioning opts_apg.constraints='soft';
opts_prcnd.dual_hessian = 1;
opts_prcnd.prcnd = 0;
optionPrecond.E = S.E;
optionPrecond.Ed = S.Ed;
optionPrecond.exact_prcnd = 0;

[dwnApgOptimModelPrecond, detailPrecond] = preconditionDwnSystemWithScenarioTree(dwnApgOptimModel, uncertainTree,...
    dwnApgOptimCost, optionPrecond);

if(solver.APG)
    dwnFactorStepPrecond = calculateDwnFactorStep(dwnApgOptimModelPrecond, dwnApgOptimCost, uncertainTree);
end
%% APG algorithm

optionApg.state = eliminateCurrentState;

%opts_apg.x=0.1*S.xmin;
optionApg.x = 0.5*(S.xmax-P.xs)+P.xs;
optionApg.E = S.E;
optionApg.Ed = S.Ed;
optionApg.constraints = 'soft';
optionApg.distance = 'yes';

optionApg.steps = 400;
optionApg.lambda = 1.3*1/detailPrecond.norm;
if(solver.APG)
    tic
    [dwnApgResult, detailsApgPrecond] = solveSmpcDwnWithApg(dwnApgOptimModelPrecond,...
        dwnFactorStepPrecond, uncertainTree, dwnApgOptimCost, optionApg);
    toc
end

%% Gurobi Algorithm 
optionGurobi.x = optionApg.x;
optionGurobi.uprev = dwnModel.L*optionApg.state.v+optionApg.state.prev_vhat;
optionGurobi.E = S.E;
optionGurobi.Ed = S.Ed;
optionGurobi.demand = optionApg.state.demand;

if(solver.gurobi)
    
    dwnGurobiModel = gurobiDwnModel(dwnApgOptimModel, uncertainTree, dwnApgOptimCost, optionGurobi);
    
    params.outputflag = 0;
    
    dwnGurobiModel = gurobiDwnUpdate(dwnGurobiModel, dwnApgOptimModelPrecond, uncertainTree,...
        dwnApgOptimCost, optionGurobi);
    tic
    gurobiResult = gurobi(dwnGurobiModel,params);
    toc
    
    if(strcmp(gurobiResult.status,'OPTIMAL'))
        control.control_time = gurobiResult.runtime;
        nu = dwnModel.nu;
        nx = dwnModel.nx;
        nvar2=nu+2*nx;
        dwnGuorbiResult.X(:,1)=optionApg.x;
        for i=1:length(uncertainTree.stage)
            dwnGuorbiResult.U(:,i) = gurobiResult.x((i-1)*nvar2+1:(i-1)*nvar2+nu);
            dwnGuorbiResult.X(:,i+1) = gurobiResult.x((i-1)*nvar2+nu+1:(i-1)*nvar2+nu+nx);
            dwnGuorbiResult.Xs(:,i) = gurobiResult.x((i-1)*nvar2+nu+nx+1:i*nvar2);
        end
        control.Z = dwnGuorbiResult;
        control.flag=1;
    else
        control.flag=0;
    end
end
%% Optimality calculation

optionGurobi.U = dwnGuorbiResult.U;
stopingCondition = calculateOptimality(dwnApgResult,detailsApgPrecond.t,...
    detailsApgPrecond.W,dwnApgOptimCost,dwnApgOptimModelPrecond,uncertainTree,optionGurobi);

max(max(abs(dwnApgResult.U(:,:) - dwnGuorbiResult.U(:,:))))/3600
max(max(abs(dwnApgResult.X(:,:) - dwnGuorbiResult.X(:,:))))
max(max(stopingCondition.norm_percentage));

gurobiObjective = calculateDwnObjective( dwnGuorbiResult, uncertainTree,...
    dwnApgOptimCost, optionGurobi);
apgObjective = calculateDwnObjective( dwnApgResult, uncertainTree,...
    dwnApgOptimCost, optionGurobi);
