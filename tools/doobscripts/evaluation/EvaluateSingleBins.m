function [thresh, cntR, sumR, cntP, sumP, VB, VO] = EvaluateSingle(pb, gt, prFile, varargin)
    % [thresh,cntR,sumR,cntP,sumP, VB, VO] = EvaluateSingle(pb, gt, prFile, varargin)
    %
    % Calculate precision/recall curve.
    %
    % INPUT
    %	pb  :       [N,M,2], prediction results
    %               the first is soft boundary map
    %               the second is occlusion orientation map
    %
    %	gt	:       Corresponding groundtruth
    %               [N,M,2*t], where t is the number of gt annotation
    %
    %   prFile  :   Temporary output for this image.
    %
    % OUTPUT
    %	thresh		Vector of threshold values.
    %	cntR,sumR	Ratio gives recall.
    %	cntP,sumP	Ratio gives precision.
    %   VB, VO      Visulization for Bounadry and Occlusion orientation results.

    opt = struct('nthresh', 99, 'maxDist', 0.0075, 'thinpb', 1, ...,
    'w_occ', 0, 'renormalize', 1, 'rank_score', [], 'debug', 0, 'fastmode', 0);
    opt = CatVarargin(opt, varargin);
    nthresh = opt.nthresh;
    maxDist = opt.maxDist;
    thinpb = opt.thinpb;
    rank_score = opt.rank_score;

    thresh = linspace(1 / (nthresh + 1), 1 - 1 / (nthresh + 1), nthresh)';
    % zero all counts
    if opt.w_occ
        cntR_occ = zeros(size(thresh));
        cntP_occ = zeros(size(thresh));
    end

    cntR = zeros(size(thresh));

    cntP = zeros(size(thresh));
    sumP = zeros(size(thresh));

    pb_num = size(pb, 3);

    if pb_num >= 2 && opt.w_occ
        occ = pb(:, :, 2);
        pb = pb(:, :, 1);
    end

    if size(gt, 3) >= 2 || opt.w_occ
        % first is gt edge, second is gt occ
        gt_occ = gt(:, :, 2:2:end);
        gt = gt(:, :, 1:2:end);
    end

    %
    if thinpb
        % perform nms operation
        pb = edge_nms(pb, 0);
        if opt.debug; imshow(1 - pb); pause; end
    end

    img_name = split(prFile, '.');
    % imwrite(mat2gray(gt), strcat(img_name{1}, '_gt.png'));
    imwrite(mat2gray(1 - pb), strcat(img_name{1}, '_nms.png'));

    if ~isempty(rank_score);
        pb(pb > 0) = rank_score(pb > 0);
    end

    if opt.renormalize
        id = pb > 0;
        pb(id) = (pb(id) - min(pb(id))) / (max(pb(id)) - min(pb(id)) + eps);
    end

    sumR = 0;
    if (nargout >= 6), VB = zeros([size(pb) 3 nthresh]); end
    if (nargout >= 7), VO = zeros([size(pb) 4 nthresh]); end

    for i = 1:size(gt, 3),
        sumR = sumR + sum(sum(gt(:, :, i)));
    end

    sumR = sumR .* ones(size(thresh));
    % match1_all = zeros(size(gt));

    % Note that: The fastmode written by Peng Wang, which
    % refer to https://github.com/pengwangucla/MatLib/blob/master/Evaluation/EvaluateBoundary/EvaluateSingle.m#L82
    % And the result will drop. You can only use to test.
    % The other statistical method modified by Guoxia Wang from
    % https://github.com/pdollar/edges/blob/master/edgesEvalImg.m#L61

    bins_num = 8;
    bins_cntR = zeros(bins_num, nthresh);
    bins_sumR = zeros(bins_num, nthresh);
    bins_cntP = zeros(bins_num, nthresh);
    bins_sumP = zeros(bins_num, nthresh);
    bins_cntR_occ = zeros(bins_num, nthresh);
    bins_cntP_occ = zeros(bins_num, nthresh);
    bin_len = 2 * pi / bins_num;

    for j = 1:bins_num
        bins_gt = gt;
        bins_gt_occ = gt_occ;
        bins_gt_occ(bins_gt_occ < -pi) = -pi;
        bins_gt_occ(bins_gt_occ > pi) = pi;
        bin_l = -pi + (j - 1) * bin_len;
        bin_r = -pi + j * bin_len;
        bins_gt = (bins_gt_occ >= bin_l) & (bins_gt_occ < bin_r) & bins_gt;

        if opt.w_occ
            cntR_occ = zeros(size(thresh));
            cntP_occ = zeros(size(thresh));
        end
        cntR = zeros(size(thresh));
        cntP = zeros(size(thresh));
        sumP = zeros(size(thresh));
        sumR = 0;
        for i = 1:size(bins_gt, 3),
            sumR = sumR + sum(sum(bins_gt(:, :, i)));
        end
        sumR = sumR .* ones(size(thresh));

        for t = 1:nthresh,
            bmap = double(pb >= max(eps, thresh(t)));
            Z = zeros(size(pb)); matchE = Z; matchG = Z; allG = Z;
            if opt.w_occ; matchO = Z; matchGO = Z; end
            n = size(bins_gt, 3);

            for i = 1:n,
                [match1, match2] = correspondPixels(bmap, double(bins_gt(:, :, i)), maxDist);

                if opt.w_occ
                    [match1_occ, match2_occ] = correspondOccPixels(match1, ...,
                    occ, bins_gt_occ(:, :, i));

                    matchO = matchO | match1_occ > 0;
                    matchGO = matchGO + double(match2_occ > 0);
                end

                matchE = matchE | match1 > 0;
                matchG = matchG + double(match2 > 0);
                allG = allG + bins_gt(:, :, i);
            end

            if opt.w_occ
                cntP_occ(t) = nnz(matchO(:));
                cntR_occ(t) = sum(matchGO(:));
            end

            sumP(t) = nnz(bmap);
            cntP(t) = nnz(matchE);
            cntR(t) = sum(matchG(:));

            % optinally create visualization of matches
            if (nargout < 6), continue; end
            cs = [1 0 0; 0 .7 0; .7 .8 1]; cs = cs - 1;
            FP = bmap - matchE; TP = matchE; FN = (allG - matchG) / n;
            for g = 1:3, VB(:, :, g, t) = max(0, 1 + FN * cs(1, g) + TP * cs(2, g) + FP * cs(3, g)); end
            VB(:, 2:end, :, t) = min(VB(:, 2:end, :, t), VB(:, 1:end - 1, :, t));
            VB(2:end, :, :, t) = min(VB(2:end, :, :, t), VB(1:end - 1, :, :, t));

            if (nargout < 7), continue; end
            % FN1 are false negative boundaries
            % FN2 are correctly labeled boundaries but incorrect occlusion
            FP = bmap - matchO; TP = matchO;
            FN1 = (allG - matchG) / n - (matchE - matchO);
            FN2 = matchE - matchO;
            VO(:, :, 1, t) = TP; VO(:, :, 2, t) = FP; VO(:, :, 3, t) = FN1; VO(:, :, 4, t) = FN2;
        end
        bins_sumR(j, :) = sumR;
        bins_cntR(j, :) = cntR;
        bins_cntP(j, :) = cntP;
        bins_sumP(j, :) = sumP;
        bins_cntR_occ(j, :) = cntR_occ;
        bins_cntP_occ(j, :) = cntP_occ;

    end
    fprintf('#');
    % tmp_len = 2 * pi / bins_num;

    % for j = 1:bins_num
    %     tmp_occ = gt_occ(:, :, 1);
    %     tmp_occ(tmp_occ < -pi) = -pi;
    %     tmp_occ(tmp_occ > pi) = pi;
    %     tmp_l = -pi + (j - 1) * tmp_len;
    %     tmp_r = -pi + j * tmp_len;
    %     tmp_ind = tmp_occ >= tmp_l & tmp_occ < tmp_r & gt(:, :, 1) > 0;

    %     bins_sumR(j, :) = sum(tmp_ind(:));
    %     % disp(size(bins_sumR(j, :)));
    %     % disp(size(sumR));
    %     % disp(size(sumP));
    %     bins_sumP(j, :) = ceil(bins_sumR(j, :) / sumR' * sumP');
    % end

    % output
    fid = fopen(prFile, 'w');

    if fid == -1
        error('Could not open file %s for writing.', prFile);
    end

    if opt.w_occ
        fprintf(fid, '%10g %10g %10g %10g %10g %10g %10g\n', [thresh cntR sumR cntP sumP, cntR_occ, cntP_occ]');
    else
        fprintf(fid, '%10g %10g %10g %10g %10g\n', [thresh cntR sumR cntP sumP]');
    end

    fclose(fid);

    % disp(occ_bins);

    tmp_ind = bins_sumP < bins_cntP;
    bins_sumP(tmp_ind) = bins_cntP(tmp_ind) + 1;

    for j = 1:bins_num
        occ_bins_file = sprintf("%s.%d_%d_bins.txt", prFile, bins_num, j);
        occ_bins_fid = fopen(occ_bins_file, 'w');
        fprintf(occ_bins_fid, '%10g %10g %10g %10g %10g %10g %10g\n', [thresh bins_cntR(j, :)' bins_sumR(j, :)' bins_cntP(j, :)' bins_sumP(j, :)', bins_cntR_occ(j, :)', bins_cntP_occ(j, :)']');
        fclose(occ_bins_fid);
    end

    % occ_bins_file = [prFile, '.occbins.txt'];
    % occ_bins_fid = fopen(occ_bins_file, 'w');
    % fprintf(occ_bins_fid, '%10g %10g\n', occ_bins');
    % fclose(occ_bins_fid);

    % bins_file = [prFile, '.edgebins.txt'];
    % bins_fid = fopen(bins_file, 'w');
    % fprintf(bins_fid, '%10g %10g\n', edge_bins');
    % fclose(bins_fid);
    clear all;
end
