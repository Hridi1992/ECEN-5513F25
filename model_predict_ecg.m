function PredClassProb = model_predict_ecg(TimeSeriesSignal) %#codegen
% This function is only intended to support wavelet deep learning examples.
% It may change or be removed in a future release.
    coder.gpu.kernelfun();
    
    % parameters
    ModFile = 'ecg_model.mat'; % file that saves neural network model
    ImgSize = [227 227]; % input image size for the ML model
    
    % Initialize output
    PredClassProb = zeros(1, 3, 'single');
    
    % Handle signal length for code generation
    % For code generation, we need to handle variable input sizes
    if length(TimeSeriesSignal) ~= 65536
        % For code generation testing, use first 1000 samples
        % In production on Jetson, this will receive 65536 samples
        sig_len = min(length(TimeSeriesSignal), 1000);
        signal_to_use = TimeSeriesSignal(1:sig_len);
        
        % Pad if shorter than 1000
        if sig_len < 1000
            signal_to_use = [signal_to_use, zeros(1, 1000 - sig_len, 'like', TimeSeriesSignal)];
        end
    else
        signal_to_use = TimeSeriesSignal;
    end
    
    %% cwt transformation for the signal
    im = cwt_ecg_jetson_ex(signal_to_use, ImgSize);
    
    %% model prediction
    persistent model;
    if isempty(model)
        model = coder.loadDeepLearningNetwork(ModFile, 'mynet');
    end

    PredClassProb = predict(model, im);
    
end
