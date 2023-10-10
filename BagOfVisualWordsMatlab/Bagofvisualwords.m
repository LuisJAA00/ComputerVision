clear all
clf
close all

wea = 'C:\Users\User\Desktop\dataset\asl_alphabet_train\asl_alphabet_train'
imds = imageDatastore(wea,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');

addpath('C:\Users\User\Desktop'); % path where function is saved (CustomSurf)

[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');

extractor = @CustomSurf4;
bag = bagOfFeatures(imds, 'CustomExtractor', extractor,'VocabularySize',1000)

img = readimage(imds,1)


categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);

confMatrix = evaluate(categoryClassifier,testSet)