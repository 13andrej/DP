function [retval, label] = runScript3 (curve_path, slope_threshold, amp_threshold, smooth_width, peak_group, smooth_type)
    reading_data = false;
    y = [];
    label = '';

    fileID = fopen(curve_path, 'r');
    while ~feof(fileID)
        line = fgetl(fileID);
        if reading_data
            y(end + 1) = str2double(strsplit(line, '\t'){2});
        elseif startsWith(line, '#')
            reading_data = true;
        elseif startsWith(line, 'Glint position:')
            # label = str2double(strcmp(strsplit(line, ' '){2}, 'True'));
            label = strsplit(line, ' '){3};
        end
    end
    fclose(fileID);

    y = y(:); % Convert to column vector
    x = 0:1/(length(y)-1):1;
    retval = findpeaksSG(x,y,[slope_threshold], amp_threshold, smooth_width, peak_group, smooth_type);
endfunction