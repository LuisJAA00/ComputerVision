function [features, featureMetrics, varargout] = CustomSurf(I)






%% Step 2: Select Point Locations for Feature Extraction
% Here, a regular spaced grid of point locations is created over I. This
% allows for dense SURF feature extraction. 

    % Process to obtain the points

    img = I;
% ILUMINACION                   ilumina la imagen
    amt = 0.75;
    img = imlocalbrighten(img,amt,AlphaBlend=true);

%%%%%%%%%%%%
% GAUSSIAN FILTER               ruido, smoth, brillo y granulado
   sigma = 2; % "level" of gaussian filter
   img = imgaussfilt(img, sigma);

%%%%%%%%%%%%%%
% MEDIAN FILTER                 smoth, brillo y granulado
    [height,width,numChannels] = size(img); % jsut in case
    if numChannels > 1
      grayImage = rgb2gray(img);
    else
       grayImage = img;
    end
   windowSize = [3, 3]; % Kernel
   img = medfilt2(grayImage, windowSize);

%%%%%%%%%%%%%%%%
% EQUALIZACION DE HISTOGRAMAS           para mejorar contraste
    img = histeq(img);
%%%%%%%%%%%%%%%%
% Check if a pixel is completely black (0) or close to it, convert it to white (255)
    img(img < 25) = 255;


% TRESHOLD 
thresholdValue = 80; % Adjust this value as needed
img = img >= thresholdValue;
%%%%%%%%%%%%%%%%
% Dilata la imagen
    % Kernel
    se = strel('disk', 5);

    % Perform dilation
    
    img = imdilate(img, se);
    img = imerode(img, se);
%%%%%%%%%%%%%%%%


points = detectSURFFeatures(img); % Obten la ubicacion (keypoints)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    
%% Step 3: Extract features
% Finally, extract features from the selected point locations. The features
% can be numeric or binary features.
[features,valid_points] = extractFeatures(grayImage,points);



%% Step 4: Compute the Feature Metric
% The feature metrics indicate the strength of each feature, where larger
% metric values are given to stronger features. The feature metrics are
% used to remove weak features before bagOfFeatures learns a visual
% vocabulary. You may use any metric that is suitable for your feature
% vectors.
%
% Use the variance of the SURF features as the feature metric.
featureMetrics = var(features,[],2);

% Alternatively, if a feature detector was used for point selection,
% the detection metric can be used. For example:
%
% featureMetrics = multiscaleSURFPoints.Metric;

% Optionally return the feature location information. The feature location
% information is used for image search applications. See the retrieveImages
% and indexImages functions.
if nargout > 2
    % Return feature location information
    varargout{1} = multiscaleGridPoints.Location;

end