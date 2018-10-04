function [q,I,S2I] = singlestage(xi,p,ni)
% Optimal scenario reduction via forward recursion - single stage
%
% Reduce scenarios to a desired number by minimizing WK distance.
%
% Input arguments:
% XI    : random variables
% P     : probabilities
% NI    : number of desired scenarios
%
% Output arguments:
%
% Q     : probabilities of reduced scenarios
% I     : indices of kept scenarios
% S2I   : S2I(j)=i iff scenario i \in I is the closest to scenario j 
%
% by D. Bernardini

%% Start scenario reduction algorithm
%
% see Algorithm 2 in Scenario tree generation for multi-stage stocahstic 
% programs Heitsch and Romisch

% Reduce scenarios
ns = size(xi,1); % no. of scenarios
S = (1:ns)'; % set of all scenarios
J = S; % set of eliminated scenarios
I = zeros(ni,1); % set of reduced scenarios
for i=1:ni % for all scenarios that we want to keep
    Dmin = inf;
    for s=1:numel(J) % for all candidate scenarios
        D = 0;
        H = setdiff(S,setdiff(J,s));
        for j=1:numel(J) % for all candidate scenarios but s
            if j~=s
                normximin = inf;
                for h=1:numel(H) % for all scenarios not in J\s
                    normxih = norm(xi(J(j),:)-xi(H(h),:));
                    if normxih < normximin
                        normximin = normxih;
                    end
                end
                D = D + p(J(j))*normximin;
            end
        end
        if D < Dmin % find the minimum D and the corresponding J index
            Dmin = D;
            smin = s;
        end
    end
    I(i) = J(smin);
    J = [J(1:(smin-1)); J((smin+1):end)];
end

% Apply redistribution rule: add probabilities of deleted scenarios to the 
% one of the closest kept scenario
q = zeros(ns,1); % probabilities after redistribution
S2I = zeros(ns,1); % S2I(j)=i iff scenario i \in I is the closest to scenario j 
for j=1:ns % for all scenarios
    dmin = inf;
    for i=1:ni
        d = norm(xi(j,:)-xi(I(i),:)); 
        if d < dmin
            dmin = d;
            imin = i;
        end
    end
    S2I(j) = I(imin);
end    

for i=1:ni
    q(I(i)) = sum(p(S2I==I(i))); % define probabilities of reduced scenarios
end

% plot results
% figure(1)
% hold on
% for i=1:ns
%     plot(xi(i,1),xi(i,2),'b.','MarkerSize',6*ns*p(i));
%     if q(i) > 0 
%         plot(xi(i,1),xi(i,2),'ro','MarkerSize',3*ns*q(i));
%     end
% end
% grid on
