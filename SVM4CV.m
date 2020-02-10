%% Cross Validation
function [est_label, score] = SVM4CV(training_data, training_label, test_data, kernel_type, sigma,c, verbose )
%% Get the inputs
[num_training_data, data_dim] = size(training_data);    %Egitim Verisinin Boyutu Aliniyor.
num_test_data = size(test_data, 1);                     %Test Verisinin Boyutu Aliniyor.
classes = unique(training_label);                       %Genel Sinif Isimleri Aliniyor.
num_classes = numel(classes);                           %Sinif Sayisi Aliniyor.
score = nan(num_test_data, num_classes);                %Her sinif ile egitim verisinin score degerini tutacak array tanimlaniyor.
svms = cell(num_classes,1);                             %Sinif Adedi kadar SVM hücresi olusturuluyor.

%% Train the svms
for k=1:num_classes
    if verbose
        fprintf(['Training Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    %Her sinif ile ayri ayri sistem egitime sokuluyor.
    class_k_label = training_label == classes(k);
        svms{k} = fitcsvm(training_data, class_k_label, 'Standardize', true, ...
            'KernelScale', sigma, 'KernelFunction', kernel_type, 'CacheSize', 'maximal' , 'BoxConstraint' , c);   
end

%% Classify the test data

for k=1:num_classes
    if verbose
        fprintf(['Classifying with Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
    end
    [~, temp_score] = predict(svms{k}, test_data);
    score(:, k) = temp_score(:, 2);
end

[~, est_label] = max(score, [], 2);

clear svms
end