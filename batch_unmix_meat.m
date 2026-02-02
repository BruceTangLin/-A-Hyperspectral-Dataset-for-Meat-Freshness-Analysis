function batch_unmix_meat(data_dir, out_dir)
%BATCH_UNMIX_MEAT Batch FCLS unmixing for meat freshness analysis
%
%   batch_unmix_meat(data_dir, out_dir)
%
%   This script performs hyperspectral linear unmixing (FCLS) on a batch of
%   reflectance-calibrated hyperspectral cubes stored as .mat files under
%   `data_dir`. The cubes are expected to be named like:
%       01.mat, 02.mat, ..., 19.mat, ...
%
%   For each cube, the script:
%     1) loads the hyperspectral cube (lines × samples × bands)
%     2) reshapes pixels into a matrix Y (bands × N)
%     3) estimates abundances via Fully Constrained Least Squares (FCLS):
%            min_a ||E a - x||^2
%            s.t.  a >= 0,  sum(a)=1
%     4) reshapes abundances back to abundance maps AbMap (lines × samples × K)
%     5) visualizes the abundance maps as ONE composite RGB image (soft mixing)
%     6) saves results (AbMap + statistics + RGB bmp)
%
%   Inputs:
%       data_dir : folder containing .mat files for each time point
%       out_dir  : output folder for saving results; if empty, a default
%                  folder "unmix_results" will be created under data_dir
%
%   Outputs (saved to out_dir for each file XX.mat):
%       - result_XX.bmp  : composite abundance visualization (soft color mixing)
%
%   Requirements:
%       - Optimization Toolbox: quadprog (called inside fcls_quadprog)
%       - Your helper functions:
%           load_mat.m, l2norm_cols.m, fcls_quadprog.m, find_first_3d_array.m
%
%   Example:
%       batch_unmix_meat('/home/data/tlrz/meat_unmixing/mat_data', ...
%                        '/home/data/tlrz/meat_unmixing/results');
%
%   Notes:
%       1) Endmembers are fixed and loaded from predefined ROI-averaged spectra.
%          This enables comparability across time points.
%       2) L2 normalization is applied to both endmembers and pixels to reduce
%          brightness/illumination variation. If your data is already normalized
%          or you prefer SNV/derivatives, you can modify this step.
%       3) FCLS with quadprog is relatively slow for very large images. For speed,
%          consider using a meat mask (maskMeat) so that only meat pixels are unmixed.
%
%   Author: Linruize Tang
%   License: MIT (or your preferred license)
%

% -----------------------------
% 0) Handle default output dir
% -----------------------------
if nargin < 2 || isempty(out_dir)
    % If output directory is not provided, create a default folder
    out_dir = fullfile(data_dir, 'unmix_results');
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% -----------------------------
% 1) Load fixed endmembers (ROI-averaged spectra)
%    Each e1..e5 should be a column vector: (bands × 1)
% -----------------------------
e1 = load_mat('lean_fresh_end.mat'); % Lean_fresh
e2 = load_mat(fat_fresh_end.mat');  % Fat_fresh
e3 = load_mat('lean_dry_end.mat');   % Lean_stale/dry
e4 = load_mat('fat_dry_end.mat');    % Fat_stale/dry
e5 = load_mat('bk_end.mat');         % Belt/background

% Endmember matrix E: (bands × K), where K=5 here.
E = cat(2, e1, e2, e3, e4, e5);  % (bands, 5)

% quadprog requires double inputs; also keep a consistent numeric type
E = double(E);

% L2 normalize endmembers (column-wise) to reduce illumination/scale effects
E = l2norm_cols(E);

K = size(E, 2);

%% -----------------------------
% 2) Define visualization colors (one color per endmember)
%    Order: [Lean_fresh, Fat_fresh, Lean_stale, Fat_stale, Belt]
% -----------------------------
C = [
    0.10 0.80 0.10;   % e1 Lean_fresh  (green)
    1.00 0.90 0.10;   % e2 Fat_fresh   (yellow)
    0.90 0.10 0.10;   % e3 Lean_stale  (red)
    0.60 0.10 0.70;   % e4 Fat_stale   (purple)
    0.20 0.40 0.90;   % e5 Belt        (blue)
];

%% -----------------------------
% 3) Collect and sort all .mat files by numeric filename (01.mat -> 1)
% -----------------------------
files = dir(fullfile(data_dir, '*.mat'));
if isempty(files)
    error('No .mat files found under: %s', data_dir);
end

% Extract numeric IDs from filenames for sorting
nums = nan(numel(files), 1);
for i = 1:numel(files)
    tok = regexp(files(i).name, '(\d+)\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        nums(i) = str2double(tok{1});
    else
        % If filename does not match the pattern, put it at the end
        nums(i) = inf;
    end
end

% Sort by numeric order
[~, order] = sort(nums);
files = files(order);

fprintf('Found %d mat files. Output -> %s\n', numel(files), out_dir);

%% -----------------------------
% 4) Process each time point / file
% -----------------------------
for fi = 1:numel(files)
    fname = files(fi).name;
    fpath = fullfile(files(fi).folder, fname);

    % Parse numeric ID for output naming (keep "01", "19", etc.)
    tok = regexp(fname, '(\d+)\.mat$', 'tokens', 'once');
    if isempty(tok)
        idStr = sprintf('%03d', fi);
    else
        idStr = sprintf('%02d', str2double(tok{1}));
    end

    fprintf('[%d/%d] Processing %s ...\n', fi, numel(files), fname);

    % -------------------------
    % 4.1 Load hyperspectral cube
    %     Expected variable name: hyper_image (lines × samples × bands)
    %     Fallback: find any 3D numeric array in the .mat file.
    % -------------------------
    S = load(fpath);

    if isfield(S, 'hyper_image')
        cube = S.hyper_image;
    else
        cube = find_first_3d_array(S);
        if isempty(cube)
            warning('Skip %s: cannot find 3D cube (lines,samples,bands).', fname);
            continue;
        end
    end

    % cube dimensions: (lines, samples, bands)
    [lines, samples, bands] = size(cube);

    % Ensure endmember bands match cube bands
    if size(E, 1) ~= bands
        warning('Skip %s: bands mismatch. E has %d bands, cube has %d bands.', ...
            fname, size(E,1), bands);
        continue;
    end

    % -------------------------
    % 4.2 Optional meat mask (for speed & cleaner statistics)
    %     If you have a per-image mask, load or compute it here.
    %     maskMeat should be logical (lines × samples).
    % -------------------------
    maskMeat = [];  % currently disabled

    if ~isempty(maskMeat)
        % Only unmix pixels within mask
        idx = find(maskMeat(:));
    else
        % Unmix all pixels
        idx = 1:(lines * samples);
    end

    % -------------------------
    % 4.3 Reshape cube -> Y (bands × N)
    %     N = lines*samples; each column is one pixel spectrum.
    % -------------------------
    Y = reshape(cube, [], bands)';  % (bands, lines*samples)
    Y = double(Y);

    % Apply the same normalization as endmembers
    Y = l2norm_cols(Y);

    % Select pixels to unmix
    Ysel = Y(:, idx);              % (bands, Nsel)

    % -------------------------
    % 4.4 FCLS unmixing (pixel-wise QP)
    %     A: (K × Nsel), each column sums to 1 and is non-negative
    % -------------------------
    A = fcls_quadprog(E, Ysel);

    % -------------------------
    % 4.5 Reshape abundances back to abundance maps
    %     AbMap: (lines × samples × K)
    % -------------------------
    Ab = zeros(K, lines * samples);
    Ab(:, idx) = A;
    AbMap = reshape(Ab', lines, samples, K);

    % -------------------------
    % 4.6 Create ONE composite RGB visualization (soft mixing)
    %     RGB = sum_k abundance_k * color_k
    %     This visualizes mixtures smoothly (not hard labels).
    % -------------------------
    RGB = zeros(lines, samples, 3);
    for k = 1:K
        RGB(:,:,1) = RGB(:,:,1) + AbMap(:,:,k) * C(k,1);
        RGB(:,:,2) = RGB(:,:,2) + AbMap(:,:,k) * C(k,2);
        RGB(:,:,3) = RGB(:,:,3) + AbMap(:,:,k) * C(k,3);
    end

    % Mask outside region as black if mask exists
    if ~isempty(maskMeat)
        for ch = 1:3
            tmp = RGB(:,:,ch);
            tmp(~maskMeat) = 0;
            RGB(:,:,ch) = tmp;
        end
    end

    % Clip to [0,1] and convert to uint8 for saving
    RGB_show  = min(max(RGB, 0), 1);
    RGB_uint8 = im2uint8(RGB_show);

    % -------------------------
    % 4.7 Save results
    % -------------------------
    outBmp = fullfile(out_dir, sprintf('result_%s.bmp', idStr));
    imwrite(RGB_uint8, outBmp);

    fprintf('  Saved: %s\n', outBmp);

end

fprintf('All done.\n');
end
