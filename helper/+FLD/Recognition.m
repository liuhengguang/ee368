function [predictions, ProjectedFisher, ProjectedPCA] = Recognition(Test_Set, m_database, V_PCA, V_Fisher, ProjectedImages_Fisher, ...
    training_labels, threshold, flip)
% Recognizing step....
%
% Description: This function compares two faces by projecting the images into facespace and 
% measuring the Euclidean distance between them.
%
% Argument:      TestSet                - Each column of TestSet is a
%                                         vectorized image
%
%                m_database             - (M*Nx1) Mean of the training database
%                                         database, which is output of 'EigenfaceCore' function.
%
%                V_PCA                  - (M*Nx(P-1)) Eigen vectors of the covariance matrix of 
%                                         the training database

%                V_Fisher               - ((P-1)x(C-1)) Largest (C-1) eigen vectors of matrix J = inv(Sw) * Sb

%                ProjectedImages_Fisher - ((C-1)xP) Training images, which
%                                         are projected onto Fisher linear space
% 
% Returns:       OutputName             - Name of the recognized image in the training database.
%
% See also: RESHAPE, STRCAT

% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir                  


% Constants 
[image_size, num_test] = size(Test_Set); 

Difference = double(Test_Set)- repmat(m_database,1,num_test);
ProjectedFisher = V_Fisher' * V_PCA' * Difference; 
ProjectedPCA = V_PCA' * Difference; 

predictions = zeros(size(ProjectedFisher)); 
if flip
    predictions(ProjectedFisher < threshold) = 2; 
    predictions(ProjectedFisher >= threshold) = 1; 
else
    predictions(ProjectedFisher < threshold) = 1; 
    predictions(ProjectedFisher >= threshold) = 2; 
end

% euclidean_distances = pdist2(ProjectedFisher', ProjectedImages_Fisher');
% [Euc_dist_min , Recognized_index] = min(euclidean_distances, [], 2);
% predictions = training_labels(Recognized_index');  




% % K closest Neighbors 
% predictions = []; 
% k = 100; 
% for i = 1:num_test
%     distances = euclidean_distances(i, :); 
%     [~, I] = sort(distances);
%     top = I(1:k); 
%     labels = training_labels(top') 
%     mode_label = mode(labels); 
%     predictions = [predictions mode_label]; 
% end



