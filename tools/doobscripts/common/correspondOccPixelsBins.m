function [bin_cntR, bin_cntR_occ] = correspondOccPixelsBins(match1, occ, gt_occ);
    % match1 is the matched pixel of bmap1 indexed with corresponding pixel
    % of bmap2
    % m1 = false(size(match1));
    % m2 = false(size(match1));
    bmap = find(match1(:) > 0);
    gt_ind = match1(bmap);

    theta = occ(bmap);
    gt_theta = gt_occ(gt_ind);

    ind = match_theta(theta, gt_theta);

    bins_num = 8;
    tmp_len = 2 * pi / bins_num;
    bin_cntR_occ = zeros(bins_num, 1);
    bin_cntR = zeros(bins_num, 1);
    for j = 1:bins_num
        tmp_occ = gt_theta;
        tmp_occ(tmp_occ < -pi) = -pi;
        tmp_occ(tmp_occ > pi) = pi;
        tmp_l = -pi + (j - 1) * tmp_len;
        tmp_r = -pi + j * tmp_len;
        tmp_ind = tmp_occ >= tmp_l & tmp_occ < tmp_r;
        tmp_match = tmp_ind & ind;

        bin_cntR(j) = sum(tmp_ind(:));
        bin_cntR_occ(j) = sum(tmp_match(:));
    end

    
    % m1(bmap(ind)) = 1;
    % m2(gt_ind(ind)) = 1;

end

function ind = match_theta(theta, gt_theta)

    abs_diff = abs(theta - gt_theta);
    abs_diff = mod(abs_diff, 2 * pi);
    ind = (abs_diff <= pi / 2 | (abs_diff >= 3 * pi / 2));

end
