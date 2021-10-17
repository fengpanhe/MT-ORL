function p = callAnalyze(eval_str, out_dir)

    eval('profile on -history -historysize 1000000000');
    eval(eval_str);
    p = profile('info');
    % function_name = strtok(eval_str, '(');
    % function_name = eval_str;
    % out_file = fullfile(out_dir, strcat(eval_str, '.txt'));

    analyze(p, '123.txt');
    
end
