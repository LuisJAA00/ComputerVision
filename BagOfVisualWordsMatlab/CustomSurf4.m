function [features, featureMetrics, varargout] = CustomSurf4(I)
    
    GrayImage = rgb2gray(I);

% TEXTURE ANALISIS
    % RANGE FILTER
        J = rangefilt(I);
     % Transfer the image to the GPU
        J = gpuArray(J);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        J = imcomplement(J);
        J = rgb2gray(J);




% TRESHOLD 
thresholdValue = 230; % Adjust this value as needed
J = J >= thresholdValue;
J = ~J;
% Dilata la imagen
    % Kernel
    se = strel('square',7);

    % Perform dilation
    
   
    
    J = imdilate(J, se);
    J = imdilate(J, se);
    se = strel('square',5);
    J = imerode(J, se);
    J = imerode(J, se);
    J = imerode(J, se);
    
%%%%%%%%%%%%%%%%
% Transfer  image back to the CPU if needed
  J = gather(J);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



points = detectSURFFeatures(J);

%%%%%%% SEGUNDA OBTENCION DE PUNTOS

 I2 = I;
    img2 = I2;
% ILUMINACION                   ilumina la imagen
    amt = 1;
    img2 = imlocalbrighten(img2,amt,AlphaBlend=true);

% Transfer the image to the GPU
  img2 = gpuArray(img2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%
% GAUSSIAN FILTER               ruido, smoth, brillo y granulado
   sigma = 2; % "level" of gaussian filter
   img2 = imgaussfilt(img2, sigma);


%%%%%%%%%%%%%%
% MEDIAN FILTER                 smoth, brillo y granulado

   windowSize = [3, 3]; % Kernel
   img2 = medfilt2(GrayImage, windowSize);


%%%%%%%%%%%%%%%%
% EQUALIZACION DE HISTOGRAMAS           para mejorar contraste
    img2 = histeq(img2);



    
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
% Check if a pixel is completely black (0) or close to it, convert it to white (255)
    img2(img2 < 40) = 255;


% TRESHOLD 
thresholdValue = 255; % Adjust this value as needed
img2 = img2 ~= thresholdValue;

%%%%%%%%%%%%%%%%
% Dilata la imagen
    % Kernel
    se = strel('square', 7);

    % Perform dilation
    
    img2 = imdilate(img2, se);
    img2 = imdilate(img2, se);
    se = strel('square',5);
    img2 = imerode(img2, se);


%%%%%%%%%%%%%%%%

% Transfer  image back to the CPU if needed
  img2 = gather(img2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

points2 = detectSURFFeatures(img2);



pointsX = [points.selectStrongest(30);points2.selectStrongest(50)];




[features,valid_points] = extractFeatures(GrayImage,pointsX);
featureMetrics = var(features,[],2);

if nargout > 2
    % Return feature location information
    varargout{1} = pointsX.Location;

end
