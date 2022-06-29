function []= export_to_file(reflectivities, file_name, number_of_frequencies, file_type) 
% reflectivities_with_noise: Reflectivities to add to the file
% file_name: Beginning of the file name
% number_of_frequencies: Total number of frequencies
% thickness: Current thickness
% file_type: Either txt or csv



file_name = strcat(file_name,".",file_type);

% Save data into files
if file_type == "txt"
    file=fopen(file_name,'w');
    for i=1: length(reflectivities)
        row = '';
        for frequency=1:number_of_frequencies
            row = row + " " + num2str(reflectivities(frequency,i));
        end
        row = row + "\n";
        fprintf(file,row);
    end
    
    fclose(file);
end


if file_type == "csv"
    
     if exist(file_name, 'file') == 2
         delete(file_name);
     end
     for i=1: length(reflectivities)
         to_write = squeeze(reflectivities(:,i)');
         writematrix(to_write,file_name,'WriteMode','append');
     end
end


end