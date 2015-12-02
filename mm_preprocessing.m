function [ output_img ] = mm_preprocessing( img )
output_img = img; 
gamma = 0.2;
normalize = 1;
    
%% Gamma correction 
% (we could use 255*imadjust(X,[],[0,1],gamma), but would add dependencies 
% to the image processing toolbox); we use our implementation
output_img = gamma_correction(output_img, [0 1], [0 255], gamma);

%% Sharpen whole image
% output_img = imsharpen(output_img, 'Amount', 3.0); 

%% Eye Sharpening
eyeXMLFile = 'res/haarcascade_eye.xml';
eye_cascade = cv.CascadeClassifier(eyeXMLFile); 
eye_boxes = eye_cascade.detect(uint8(output_img));
box_sizes = []; 
sharpening_const = 3.0; 
for i = 1:numel(eye_boxes)
    box = eye_boxes{i}; 
    box_sizes = [box_sizes, box(3) * box(4)];  
end

[~, I] = sort(box_sizes, 'descend'); 

% Eye Detected - Processing 
if numel(I) >= 2
    fbox = round(eye_boxes{I(1)}); 
    sbox = round(eye_boxes{I(2)});
    fbox(3:4) = fbox(3:4) - 5; 
    sbox(3:4) = sbox(3:4) - 5;
    fbox(1:2) = fbox(1:2) + 2; 
    sbox(1:2) = sbox(1:2) + 2; 

    eye1 = output_img(fbox(2):fbox(2) + fbox(4), fbox(1): fbox(1) + fbox(3)); 
    eye2 = output_img(sbox(2):sbox(2) + sbox(4), sbox(1): sbox(1) + sbox(3)); 

    % Sharpen Areas
    eye1 = imsharpen(eye1, 'Amount', sharpening_const); 
    eye2 = imsharpen(eye2, 'Amount', sharpening_const); 

    output_img(fbox(2):fbox(2) + fbox(4), fbox(1): fbox(1) + fbox(3)) = eye1; 
    output_img(sbox(2):sbox(2) + sbox(4), sbox(1): sbox(1) + sbox(3)) = eye2; 
end

%% Mouth Sharpening 

mbox = [50, 155, 105, 40]; 
% output_img = cv.rectangle(output_img, box, ...
%         'Color', [255 0 0], 'Thickness', 2); 
% imshow(output_img, []); 
% pause()
mouth = output_img(mbox(2):mbox(2) + mbox(4), mbox(1): mbox(1) + mbox(3));
mouth = imsharpen(mouth, 'Amount', sharpening_const*2); 

output_img(mbox(2):mbox(2) + mbox(4), mbox(1): mbox(1) + mbox(3)) = mouth; 

%% Filtering 
% output_img = dog(output_img, 1, 2, 0);

%% Postprocessing
output_img = robust_postprocessor(output_img);

%% Normalization to 8bits
if normalize ~= 0
    output_img = normalize8(output_img);  
end

