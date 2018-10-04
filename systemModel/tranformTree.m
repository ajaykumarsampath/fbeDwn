function [ New_tree,options] = tranformTree(Tree )
% This function transform the tree generated into a new tree 
% structure with nodes labeled in increasing order. 
%
% INPUT-----   Tree    :    old tree
% 
% OUTPUT----  New_tree :    New tree 
%             options  :    scenario_details: contains the corespondence
%             between new tree and old tree 
%%
Nd=size(Tree.stage,1);
nw=size(Tree.value,2);
ns=size(Tree.leaves,1);


New_tree.stage=zeros(Nd,1);
New_tree.value=zeros(Nd,nw);
New_tree.prob=zeros(Nd,1);
New_tree.ancestor=zeros(Nd,1);
New_tree.children=cell(Nd-ns,1);
New_tree.leaves=zeros(ns,1);

SI=scenario_index(Tree);
Np=size(SI{1},1);
q=0;
New_tree.prob(1,1)=1;
for k=1:Np
    leaves=[];
    nodes_stage=[1];
    kk=k;
    while(kk>1)
        for kkk=1:length(nodes_stage)
            leaves=[leaves;Tree.children{nodes_stage(kkk)}];
        end
        if(kk>0)
            nodes_stage=leaves;
            leaves=[];
        end
        kk=kk-1;
    end
    no_nodes=length(nodes_stage);
    
    New_tree.stage(q+1:q+no_nodes,1)=(k-1)*ones(no_nodes,1);
    if(k==Np)
        New_tree.leaves=[q+1:q+no_nodes]';
    end
    %
    p=0;
    for j=1:no_nodes
        if(k>1)
            nodes_ancestor=find(New_tree.stage(1:q,1)==k-2);
            for kk=1:length(nodes_ancestor)
                if(find(New_tree.children{nodes_ancestor(kk)}==q+j))
                    New_tree.ancestor(q+j,1)=nodes_ancestor(kk);
                end
            end
        else
            New_tree.ancestor(q+j,1)=Tree.ancestor(nodes_stage(j),1);
        end
        if(k<Np)
            ll=q+no_nodes+p+1:q+no_nodes+p+length(Tree.children{nodes_stage(j),1});
            New_tree.children{q+j,1}=[q+no_nodes+p+1:q+no_nodes+p+length(Tree.children{nodes_stage(j),1})]';
            New_tree.value(ll,:)=Tree.value(Tree.children{nodes_stage(j),1},:);
            New_tree.prob(ll,1)=Tree.prob(Tree.children{nodes_stage(j),1},:);
            p=p+length(Tree.children{nodes_stage(j),1});
        end
    end
 %}
    q=q+length(nodes_stage);
end
%{
snodes(:,2)=snodes(:,1);

for i=1:length(Tree.children)
    nchild=Tree.children{i};
    if(~isempty(nchild)) 
        s=Tree.stage(i);
        snodes(s+1,1)=snodes(s+1,1)+1;
        New_tree.children{snodes(s+1,1),1}=snodes(s+2,2)+1:snodes(s+2,2)+length(nchild);
        snodes(s+2,2)=snodes(s+2,2)+length(nchild);
    end
end

%}

SI_new=scenario_index(New_tree);
%
sc=zeros(ns,1);
clear value
for j=1:ns
    value(:,1)=Tree.value(SI{j},1);
    for l=1:ns
        value(:,2)=New_tree.value(SI_new{l},1);
        if(max(abs(value(:,1)-value(:,2)))==0)
            sc(j,1)=l;
        end
    end 
end
options.scenario_order=sc;
%{
for i=1:ns
    value(:,1)=New_tree.value(SI_new{sc(1)},3);
    value(:,2)=Tree.value(SI{1},3);
    plot(value(:,1)-value(:,2));
    hold all;
end
%}
end

