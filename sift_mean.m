%% EE368 Project
close all; clear all; clc;
emotions = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'};  
subjectNames = './sift_test_data/*0*'; 
num_subjects = 13; 
num_emotions = 6; 

condIntensity = [0, 1, 5, 10, 15]; 
all_repeatability = zeros(num_emotions, numel(condIntensity)); 

for emotion_index = 1:6
    emotion = emotions{emotion_index};
    repeatability_scores = zeros(1, numel(condIntensity));
    allSubjects = dir(subjectNames); 
    for subject_id = 1:numel(allSubjects); 
        subjectFolder = allSubjects(subject_id).name; 
        imageFiles = {['sift_test_data/' subjectFolder '/' emotion '_SUN_3.png']};
        condImages = cell(5);
        
        for i = 1:numel(condIntensity)
           condImages{i} =  ['sift_test_data/' subjectFolder '/' emotion '_SUN_' ...
               num2str(condIntensity(i)) '.png'];
        end
        for nImage = 1:length(imageFiles)
            % Load original image
            [imgOrig, box] = preprocess(rgb2gray(imread(imageFiles{nImage})));
            [height, width] = size(imgOrig);
            x_center = width/2;
            y_center = height/2;

            % Extract SIFT keypoints on original image
            peakThresh = 4;
            edgeThresh = 10;
            [fc, dc] = vl_sift(single(imgOrig));
            xc = fc(1,:);
            yc = fc(2,:);
            numFeatures = size(fc,2);
%             disp(['Features = ' num2str(numFeatures)]);

            % Compare to different illumination conditions
            illumInt = 0:1:length(condImages)-1;
            numFeatureMatches = zeros(1, numel(illumInt));
            for nCond = 1:length(condImages)
                % Displays File Information
                display(condImages{nCond}); 

                % Load conditioned image
                [condImg, ~] = preprocess(rgb2gray(imread(condImages{nCond})), box);

                % Find SIFT keypoints on conditioned imagec
                [fc_cond, dc_cond] = vl_sift(single(condImg));
                numFeaturesCond = size(fc_cond,2);
                xc_cond = fc_cond(1,:);
                yc_cond = fc_cond(2,:);
%                 disp(['Condition = ' num2str(nCond) ', Features = ' num2str(numFeaturesCond)]);

                isFound = zeros(1, numel(xc)); 
                % Find feature matches to original image
                for nOrig = 1:length(xc)
                    matchFound = 0;
                    for nTrans = 1:length(xc_cond)
                        threshold = 2;
                        delta_x = abs(xc(nOrig)-xc_cond(nTrans));
                        delta_y = abs(yc(nOrig)-yc_cond(nTrans));
                        if ( delta_x <= threshold ) && ( delta_y <= threshold )
                            matchFound = 1;
                            break;
                        end
                    end % nTrans
                    if matchFound == 1
                        numFeatureMatches(nCond) = numFeatureMatches(nCond) + 1;
                        isFound(nOrig) = 1; 
                    end
                end % nOrig
%                 disp(['Feature Matches = ' num2str(numFeatureMatches(nCond))]);

                isFound = logical(isFound); 
%                 % Plots points have matches
%                 imshow(imgOrig, []); 
%                 hold on 
%                 plot(xc(isFound), yc(isFound), 'g.'); 
%                 plot(xc(~isFound), yc(~isFound), 'r.'); 
%                 hold off 
%                 %  sift_match(imgOrig, condImg);
%                 pause;

            end % nCond
            repeatability_scores = repeatability_scores + double(numFeatureMatches)/numFeatures; 
        end % nImages
    end % subject
    repeatability_scores = repeatability_scores/num_subjects;
    all_repeatability(emotion_index, :) = repeatability_scores; 
end % emotion

% Plot repeatability against condition
figure(nImage); clf; set(gcf, 'Position', [50 50 400 300]);
h = plot(repmat(illumInt',1, num_emotions), (all_repeatability)'); grid on;
set(h, 'MarkerFaceColor', 'b');
xlabel('Condition'); ylabel('Repeatability');
title(['Robustness to Illumination/Shading: No Preprocessing']);
legend('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'); 
axis([0 4 0 1]);
