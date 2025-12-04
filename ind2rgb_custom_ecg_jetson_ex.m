function rgb = ind2rgb_custom_ecg_jetson_ex(im, cmap)
% Custom ind2rgb function for code generation
coder.gpu.kernelfun();

% Convert indexed image to RGB
rgb = zeros(size(im, 1), size(im, 2), 3, 'single');

for i = 1:size(im, 1)
    for j = 1:size(im, 2)
        idx = double(im(i, j));
        if idx < 1
            idx = 1;
        elseif idx > size(cmap, 1)
            idx = size(cmap, 1);
        end
        rgb(i, j, 1) = single(cmap(idx, 1));
        rgb(i, j, 2) = single(cmap(idx, 2));
        rgb(i, j, 3) = single(cmap(idx, 3));
    end
end
end
