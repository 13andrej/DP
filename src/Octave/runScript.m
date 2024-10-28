# idk
l = 8;

function [features, label] = load_curve(curve_path)
    reading_data = false;
    features = [];
    label = [];

    fileID = fopen(curve_path, 'r');
    while ~feof(fileID)
        line = fgetl(fileID);
        if reading_data
            features(end + 1) = str2double(strsplit(line, '\t'){2});
        elseif startsWith(line, '#')
            reading_data = true;
        elseif startsWith(line, 'Glint:')
            label = str2double(strcmp(strsplit(line, ' '){2}, 'True'));
        end
    end
    fclose(fileID);

    features = features(:); % Convert to column vector
endfunction;


[y label] = load_curve('C:\Users\13and\PycharmProjects\DP\data\dataset\exports\5.txt');
x = 0:1/(length(y)-1):1;
plot(x,y,'c.');

P=findpeaksSG(x,y,[0.05], 20, 5, 10, 3);
disp(P);
