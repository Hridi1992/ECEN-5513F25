function im = cwt_ecg_jetson_ex(TimeSeriesSignal, ImgSize) %#codegen
% This function is only intended to support wavelet deep learning examples.
% It may change or be removed in a future release.

coder.gpu.kernelfun();

%% Create Scalogram
cfs = cwt(TimeSeriesSignal, 'morse', 1, 'VoicesPerOctave', 12);
cfs = abs(cfs);

%% Image generation
% Load colormap
cmapj128_struct = coder.load('cmapj128.mat');
cmapj128 = cmapj128_struct.cmapj128;

% Rescale and convert
cfs_rescaled = rescale(cfs);
im_indexed = round(255 * cfs_rescaled) + 1;
im_indexed = cast(im_indexed, 'uint8');  % Explicit cast

% Convert indexed to RGB
imx = ind2rgb_custom_ecg_jetson_ex(im_indexed, cmapj128);

% resize to proper size and convert to uint8 data type
im = im2uint8(imresize(imx, ImgSize)); 

end
