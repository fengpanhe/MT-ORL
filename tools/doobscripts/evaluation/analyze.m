%{
% Filename: analyze.m
% Project: functionAnalyze
% Created Date: Sunday March 31st 2019
% Author: microhe
% -----
% Last Modified:
% Modified By:
% -----
% Copyright (c) 2019 microhe
%}
function analyze(p, out_file)
    ignore_function_id = [];

    for i = 1:numel(p.FunctionTable)

        if ~strcmp(p.FunctionTable(i).CompleteName(1:4), '/run')
            ignore_function_id(end + 1) = i;
        end

    end

    analyze_res = analyze_mex(int32(p.FunctionHistory), int32(ignore_function_id));
    f = fopen(out_file, 'wt');

    for i = 1:size(analyze_res, 2)
        depth = analyze_res(1, i);
        function_id = analyze_res(2, i);
        call_times = int32(analyze_res(3, i));

        name = p.FunctionTable(function_id).FunctionName;
        file_name = p.FunctionTable(function_id).CompleteName;

        tmp_str = join(repelem("|     ", depth), '');
        fprintf(f, '%s\n', tmp_str{1});
        tmp_str = join(repelem("|     ", depth), '');
        fprintf(f, '%s____ %s  %d  %s\n', tmp_str{1}(1:end - 5), name, call_times, file_name(40:end));
    end

    fclose(f);
end
