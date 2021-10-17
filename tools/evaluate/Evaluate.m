function res = Evaluate(oriResPath, DataSet, w_occ, maxDist)
    disp(oriResPath);
    disp(DataSet);
    % currentPath = fileparts(mfilename('fullpath'));
    addpath(genpath('tools/doobscripts'));

    % DataSet = 'PIOD';
    % DataSet = 'BSDSownership';
    switch DataSet
        case 'PIOD'
            oriGtPath = 'data/PIOD/Data'
            testIdsFilename = 'data/PIOD/val_doc_2010.txt';

            opt.method_folder = 'OPNet';
            opt.model_name = 'PIOD';
        case 'BSDSownership'
            oriGtPath = 'data/BSDSownership/testfg/';
            testIdsFilename = 'data/BSDSownership/Augmentation/test_ori_iids.lst';

            opt.method_folder = 'OPNet';
            opt.model_name = 'BSDSownership';
    end

    ImageList = textread(testIdsFilename, '%s');
    opt.validate = 0;

    if opt.validate
        valid_num = 10;
        ImageList = ImageList(1:valid_num);
    end

    tic;
    respath = [oriResPath, '/'];
    evalPath = [oriResPath, '/eval_fig/']; if~exist(evalPath, 'dir') mkdir(evalPath); end

    opt.DataSet = DataSet;
    opt.maxDist = maxDist;
    opt.vis = 1;
    opt.print = 0;
    opt.overwrite = 1;
    opt.visall = 0;
    opt.append = '';
    opt.occ_scale = 1; % set which scale output for occlusion
    opt.w_occ = w_occ;
    if opt.w_occ; opt.append = '_occ'; end
    opt.scale_id = 0;

    if opt.scale_id ~= 0;
        opt.append = [opt.append, '_', num2str(opt.scale_id)];
    end

    opt.outDir = respath;
    opt.resPath = respath;
    opt.gtPath = oriGtPath;
    opt.nthresh = 99; % threshold to calculate precision and recall
    % it set to 33 in DOC for save runtime but 99 in DOOBNet.
    opt.thinpb = 1; % thinpb means performing nms operation before evaluation.
    opt.renormalize = 0;
    opt.fastmode = 0; % see EvaluateSingle.m

    if (~isfield(opt, 'method') || isempty(opt.method)), opt.method = opt.method_folder; end
    fprintf('Starting evaluate %s %s, model: %s and %s\n', DataSet, opt.method, opt.model_name, opt.append);

    EvaluateOcclusion(ImageList, opt);

    if opt.vis
        close all;

        if strfind(opt.append, '_occ');
            app_name = opt.append;

            opt.eval_item_name = 'Boundary';
            opt.append = [app_name, '_e'];
            boundary_res = plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge');

            opt.eval_item_name = 'Orientation PR';
            opt.append = [app_name, '_poc'];
            ori_res = plot_multi_eval_v2(opt.outDir, opt, opt.method); title('PRO');

            opt.append = [app_name, '_aoc'];
            ori_accuracy_res = plot_multi_occ_acc_eval(opt.outDir, opt, opt.method);
            res = [boundary_res; ori_res; ori_accuracy_res];

            bins_num = 8;

            for j = 1:bins_num
                opt.eval_item_name = convertStringsToChars(sprintf("%d_%d", bins_num, j));
                opt.append = convertStringsToChars(sprintf("_occ.txt.%d_%d_bins_poc", bins_num, j));
                plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge');
            end

        else
            opt.eval_item_name = 'Boundary';
            boundary_res = plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge');
            res = [boundary_res; boundary_res; boundary_res];
        end

    end

    toc;
end
