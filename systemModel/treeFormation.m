function [ Tree,details] = treeFormation( W,ops)
%This function generates the scenario tree.
%The no of scenatios are reduced by forward selection.
%The input is scenario of random variables

%syntax: [Tree]=Tree_formation(W,ops)
%Input:     W: The stochastic varible.
%              W=[scenarios,no of varibles,future steps]
%              The stocastic variable is considered independent
%         ops: This contains the option to the tree
%           N: prediction horizon
%          ni: branching factor 
%          Wscaling: 0/1 do/don't scale the distribution
%        Wfiltering: 0/1 do/don't filter disturbance W by system dynamics and u=Kx
%              Tmax: max number of scenarios that are passed for tree
%                    formation
%             nScen: No of scenaratios considered 
%                nw: lenght of the stochastic variable 
%               fig: 0/1 option to plot the tree 
%
%Output:  Tree: Scenario tree. It is a structre with identifiers
%              stage:Indicate at what prediction stage is the node located
%              value:The value of stochastic variable at eaach node
%               prob:The probability of the node
%           ancester:The ancester node of each of the node
%           children:The child node of each node
%             leaves:The leaf nodes of the tree

default_options = struct('N',10,'ni', 1*ones(10,1),...
    'Wscaling',0,'Wfiltering',0,'fig',0);

flds=fieldnames(default_options);
for i=1:numel(flds)
    if ~isfield(ops,flds(i))
        ops.(flds{i})=default_options.(flds{i});
    end
end

if ~isfield('P',ops)
    ops.P = ones(ops.nScen,ops.Tmax)/ops.nScen; % probability distribution of w
end

Wscaled=zeros(size(W));
if ops.Wscaling==1
    ops.Wscaled = zeros(size(W));
    for i=1:ops.nw
        Ww = W(:,i,:);
        if (std(Ww(:)) == 0)
            Wscaled(:,i,:) = W(:,i,:);
        else 
            Wscaled(:,i,:) = (W(:,i,:)-mean(Ww(:)))/std(Ww(:));
        end 
    end
else
    Wscaled = W;
end
tic
t=0;
Tree{t+1} = struct('stage',0,'value',zeros(1,ops.nw),'prob',1,...
    'ancestor',0,'children',cell(1,1),'leaves',1);

% leaves is a dynamic field: it contains the index of nodes which are leaves
% during the tree construction. Root node is 1. Other nodes are indexed
% progressively as they are added to the tree.

Cluster = cell(1,1);
Cluster{1} = (1:ops.nScen)'; % scenarios to consider for node n

N = min(ops.N,size(W,3)); % modify N to take into account availability of data

while any(Tree{t+1}.stage(Tree{t+1}.leaves) < N)
    current_number_of_leaves = numel(Tree{t+1}.leaves);
    
    for l=1:current_number_of_leaves % for each leaf node
        k = Tree{t+1}.stage(Tree{t+1}.leaves(l)); % time stage of the current node
        if k < N % leaves of this node must be evaluated
            current_number_of_nodes = numel(Tree{t+1}.ancestor);
            % select the current node
            n = Tree{t+1}.leaves(l);
            % cut this node from leaves
            Tree{t+1}.leaves = [Tree{t+1}.leaves(1:l-1); Tree{t+1}.leaves(l+1:end)];
            if isempty(Tree{t+1}.leaves), Tree{t+1}.leaves = []; end;
            % and create its leaves:
            % reduce scenarios
            % xi = w
            xi = W(Cluster{n},:,t+k+1); % data at the current stage
            xis = Wscaled(Cluster{n},:,t+k+1); % scaled data at the current stage
            ns = size(xi,1); % no. of scenarios
            p = ops.P(Cluster{n},t+k+1); % probabilities
            p = p/sum(p); % rescale p
            
            if ns > ops.ni(k+1)
                % Apply single stage scenario reduction
                [q,I,S2I] = singlestage(xis,p,ops.ni(k+1)); % use the scaled value in scenario reduction
            else
                % No need to reduce scenarios
                q = p;
                I = 1:ns;
                S2I = I;
            end
            new_nodes = numel(I);
            
            % Cluster deleted scenarios
            for i=1:new_nodes
                Cluster{current_number_of_nodes + i} = Cluster{n}(S2I==I(i));
            end
            
            % Update Tree
            Tree{t+1}.leaves(end+(1:new_nodes),1) =  current_number_of_nodes + (1:new_nodes)';
            Tree{t+1}.stage(end+(1:new_nodes),1) = k+1;
            Tree{t+1}.ancestor(end+(1:new_nodes),1) = n;
            Tree{t+1}.children{n,1} = current_number_of_nodes + (1:new_nodes)';
            Tree{t+1}.prob(end+(1:new_nodes),1) = Tree{t+1}.prob(n) * q(I);
            Tree{t+1}.value(end+(1:new_nodes),:) = xi(I,:); % this is the non scaled value
            
        end
    end
end

if(ops.fig)
    figure
    hold on;
    treeplot(Tree{t+1}.ancestor','r.','b')
    xlabel(['time ' num2str(t)]);
end
%}
details.time=toc;
details.cluster=Cluster;
end

