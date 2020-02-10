%% Cross Validation'suz hali.
clear; clc; close all;

%% Params
% User Params
perc_rate = 0.05;    %Egitim verisinden seçilecek verinin orani % default 0.05
verbose = true;
kernel_type = 'RBF';

%% Load Images
hyperdata = load('PaviaU.mat');
hyperdata = hyperdata.paviaU;

hyperdata_gt = load('PaviaU_gt.mat');
hyperdata_gt = hyperdata_gt.paviaU_gt;

%% Normalization
hyperdata = ( hyperdata - min(min(min(hyperdata))) ) /  ( max(max(max(hyperdata))) - min(min(min(hyperdata))) );
[h,w,spec_size] = size(hyperdata);
% data2vector
hypervector = reshape(hyperdata , [h*w,spec_size]);

%% 	Variables

Mask = zeros(size(hyperdata_gt));       % Egitime girecek veri
NumOfClass = max(hyperdata_gt(:));      % Hyper imge'deki class sayisi
NumOfClassElements = zeros(1,9);        % Her class'a ait etiket sayisini tutacak degisken

%% Calculate each class's elements number

for i = 1 : 1 : NumOfClass
    NumOfClassElements(i) = double(sum(sum((hyperdata_gt == i))));
end

%% ** %Perc_Rate data will select for training vector ** 
for i = 1 : 1 : NumOfClass
    perc_5 = floor(NumOfClassElements(i) * perc_rate);
    [row,col] = find(hyperdata_gt == i);
    
    %Maskedeki ilgili etiket degeri yüzde 5'e ulasmadigi sürece örnek al
    while(sum(sum(Mask == i)) ~= perc_5) 
        x = floor((rand() * (NumOfClassElements(i) - 1)) + 1);
        Mask(row(x),col(x)) = hyperdata_gt(row(x),col(x));
    end
end

%% Class's Nums

ClassSayisi = 0;
for i = 1 : 1 : NumOfClass
   ClassSayisi = ClassSayisi + sum(sum(Mask == i)); 
end

%% Created Traind and Label Vector

[trainingData_row,trainingData_col,values] = find(Mask ~= 0);
trainingVector = zeros(ClassSayisi,103);
trainingVectorLabel = zeros(ClassSayisi,1);

% ****  Training Vector & Training Vector Label  ****

for i = 1 : ClassSayisi
    trainingVector(i,:)      = hyperdata(trainingData_row(i),trainingData_col(i),:);
    trainingVectorLabel(i,1) = hyperdata_gt(trainingData_row(i),trainingData_col(i));
end

%% Train

%% ****************************SVM TRAIN***************************
tic;
classes = unique(trainingVectorLabel);
num_classes = numel(classes);
svms = cell(num_classes,1);

for k=1:NumOfClass
    if verbose
        fprintf(['Training Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    class_k_label = trainingVectorLabel == classes(k);
    svms{k} = fitcsvm(trainingVector, class_k_label, 'Standardize',...
        true,'KernelScale', 'auto', 'KernelFunction', kernel_type, ...
        'CacheSize', 'maximal', 'BoxConstraint', 10);
end

%**********************Classify the test data**********************
for k=1:NumOfClass
    if verbose
        fprintf(['Classifying with Classifier ', num2str(classes(k)),...
            ' of ', num2str(num_classes), '\n']);
    end
    [~, temp_score] = predict(svms{k}, hypervector);
    score(:, k) = temp_score(:, 2);                     %Her satirin ilgili sutununa sinifla ilgili score degerini diz.
end
[~, est_label] = max(score, [], 2);
prediction_svm = im2uint8(zeros(h*w, 1));

for k=1:num_classes
    prediction_svm(find(est_label==k),:) = k;
end
prediction_svm = reshape(prediction_svm, [h, w, 1]);

z = find(hyperdata_gt == 0);
prediction_svm(z) = 0;


ERR = sum(sum( (prediction_svm ~= hyperdata_gt) ));
NumOfElements = sum(sum(NumOfClassElements(:)));
NumOfTrueElements = NumOfElements - ERR;
RATE = (NumOfTrueElements / NumOfElements)*100;

%% Results
fprintf(['\n','***********************************************************','\n']);
fprintf(['\tBasari Orani : ', num2str(RATE),'\n']);
fprintf(['***********************************************************','\n']);

figure , imshow(label2rgb(prediction_svm, @jet, [.5 .5 .5])) , title('Sonuc');
figure , imshow(label2rgb(hyperdata_gt, @jet, [.5 .5 .5])) , title('Orijinal');