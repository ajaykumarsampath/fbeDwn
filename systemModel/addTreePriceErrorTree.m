function [ newTree ] = addTreePriceErrorTree( priceError, oldTree, treeOpts)
% addtreePriceErrorTree summary of this function goes here
% Detailed explanation goes here
%
%  INPUT : 
%         Tree     :   tree structure
%         optsTree :   options of the tree 
%
%  OUTPUT :
%         newTree  :   new tree structure 
%

W = zeros(treeOpts.nScen, 1, treeOpts.N);
for i=1:treeOpts.N
    W(:,:,i) = kron(-1, priceError(:, i));
end

priceOldTree = Tree_formation(W, treeOpts);
%oldTree{1}.value = kron(ratioDemand', oldTree{1}.value);
priceOldTree = priceOldTree{1};
priceTree = Tranform_tree(priceOldTree);

for iStage = 1: treeOpts.N + 1
    nodeStage = find( priceTree.stage == iStage - 1);
    for iNode = 1: length(nodeStage)
        
    end
end

end

