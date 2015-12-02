function [ processed_img, box] = preprocess( img , bounding_box)

% Input:    img - black and white image
% Output:   processed_img - black and white processed img

% Grayscale conversion 
processed_img = img; 
if(ndims(img)==3)
    processed_img = rgb2gray(img); 
end

% The size of image that will be returned 
img_size = [200, 200]; 

if nargin == 1
    xmlfile = fullfile(mexopencv.root(),'test','haarcascade_frontalface_alt2.xml');
    face_cascade = cv.CascadeClassifier(xmlfile); 
    boxes = face_cascade.detect(uint8(img));
    box = boxes{1};  
else
    box = bounding_box; 
end
% Detect the Face

x = box(1);
y = box(2); 
length = box(3); 

% Crop out the face
processed_img = processed_img(y:y+length, x:x+length); 

% Scale 
processed_img = imresize(processed_img, img_size); 

% Tan Triggs 
% processed_img = tantriggs(processed_img); 

% MM Preprocessing


end

