clear; clc; close all;
%% Info
name = 'Burak';
surname = 'Büyükyüksel';
no = '150208060';

disp('******************************');
disp(['FirstName : ',name]);
disp(['SurName   : ',surname]);
disp(['No        : ',no]);
disp('******************************');

%% Params
% User Params
perc_rate = 0.05;   %Egitim verisinden seçilecek verinin orani % default 0.05
numFolds = 5;    	%Default : 5

%System Params
intensity = 2;      %Default Val : 2
verbose = 1;
kernel_type = 'RBF';
do_cv = true;

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

%  ****  Trainining input parameter  ****

classes = unique(trainingVectorLabel);              %Training Vector Class'daki Etiket Degerleri Aliniyor
num_classes = numel(classes);                       %Etiket Degerlerinin Sayisi Aliniyor
svms = cell(num_classes);                           %Etiket Degerlerinin Sayisi Kadar Struct Olusturuluyor
num_test_data = size(hypervector,1);                %Reshape edilmis data vektoru icerisindeki eleman sayisi aliniyor.
score = nan(num_test_data , num_classes);           %Not a Number olacak sekilde array olusturuluyor.

%% Cross - Validation

if do_cv                                            %Cross Validation Yapilsin mi?
sigma = 2 .^(-5 : intensity : 5);   
C = 2 .^(-5 : intensity : 5);

rng(1);
cv_idx = [];
for k=1:num_classes
    cv_idx = [cv_idx; crossvalind('Kfold', sum(trainingVectorLabel == k),numFolds)];  %Egitim vektorleri rastgele alt gruplara ayriliyor.
end

actual_label = [];
for k=1:numFolds
    test = (cv_idx == k);
    actual_label = [actual_label; trainingVectorLabel(test)];                        % Rastgele gruplandirilan egitim verilerinin etiket degerleri aliniyor.
end

err = nan(numFolds,1);
mean_err = zeros(length(C),length(sigma));

tic;
fprintf(['\n#Cross Validation is started!','\n\n']);

for j = 1 : 1 : length(C)
    for i = 1 : 1 : length(sigma)
        for K=1:numFolds
            fprintf([num2str(i),'/',num2str(j),' [C/Sig] ','Fold ', ...
                num2str(K) ' of ', num2str(numFolds), '\n']);
            
            test = (cv_idx == K);
            train = ~test;
            label_hat = SVM4CV( ...
                            trainingVector(train, :),...
                            trainingVectorLabel(train),...
                            trainingVector(test,:),...
                            'RBF',sigma(i), C(j),  false);
            err(K) = 100*mean(label_hat ~= trainingVectorLabel(test));
        end
        mean_err(j,i) = mean(err);
    end
end
min_err = min(mean_err(:));
[index_x , index_y] = find(mean_err==min_err);          %Hata degeri min yapan C ve Sig parametrelerini sec.
c = C(index_x);
sig = sigma(index_y);
end

timeOfCV_raw = toc;
fprintf(['\n\n','CV time : ', num2str(timeOfCV_raw),'\n\n']);

%% ****************************SVM TRAIN***************************
tic;
for k=1:NumOfClass
    if verbose
        fprintf(['Training Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    class_k_label = trainingVectorLabel == classes(k);
    svms{k} = fitcsvm(trainingVector, class_k_label, 'Standardize',...
        true,'KernelScale', sig, 'KernelFunction', kernel_type, ...
        'CacheSize', 'maximal', 'BoxConstraint', c);
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

clc;
timeOfPre_raw = toc;
timeOfPre_min = floor(timeOfPre_raw / 60);
timeOfPre_sec = mod(timeOfPre_raw,60);
timeOfCV_min = floor(timeOfCV_raw / 60);
timeOfCV_sec = mod(timeOfCV_raw,60);

ERR = sum(sum( (prediction_svm ~= hyperdata_gt) ));
NumOfElements = sum(sum(NumOfClassElements(:)));
NumOfTrueElements = NumOfElements - ERR;
RATE = (NumOfTrueElements / NumOfElements)*100;

%% Results
fprintf(['\n','##***********************************************************##','\n']);
fprintf(['\tCV time         : ', num2str(timeOfCV_min) ,'[m] :: ',num2str(ceil(timeOfCV_sec)),'[s]\n']);
fprintf(['\tPrediction time : ', num2str(timeOfPre_min),'[m] :: ',num2str(ceil(timeOfPre_sec)),'[s]\n']);
fprintf(['##***********************************************************##','\n']);
fprintf(['\tThe Rate Value is : ', num2str(RATE),'\n']);
fprintf(['\tValue of Sigma    : ', num2str(sig),'\n']);
fprintf(['\tValue of C        : ', num2str(c),'\n']);

fprintf(['##***********************************************************##','\n']);

figure , imshow(label2rgb(prediction_svm, @jet, [.5 .5 .5])) , title('SVM RESULT');
figure , imshow(label2rgb(hyperdata_gt, @jet, [.5 .5 .5])) , title('SVM ORG');