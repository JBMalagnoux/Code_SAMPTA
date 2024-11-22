function [total_tests_failed,list_of_failed_tests]=test_all_libphaseret(prec,varargin)

definput.keyvals.tests=[];
definput.keyvals.ignore={};
[flags,kv]=ltfatarghelper({'tests'},definput,varargin);

tests_todo =  arrayfun(@(dEl) dEl.name(6:end-2),dir('./test_libphaseret_*.m'),'UniformOutput',0);

if ~isempty(kv.tests)
   if ischar(kv.tests)
     kv.tests = {kv.tests};
   end
   tests_todo =  kv.tests;
end

if ~isempty(kv.ignore)
    if ischar(kv.ignore)
        kv.ignore = {kv.ignore};
    end
    if ~iscell(kv.ignore)
        error('%s: Ignored tests list is incorrect.',upper(mfilename));
    end
        
    ignoreList = [];
    ignoreUsed = [];
    ignoreCell = kv.ignore;
    for ii=1:numel(tests_todo)
       res = cellfun(@(iEl) strcmpi(tests_todo{ii},iEl) , ignoreCell);
       if any(res)
           ignoreList(end+1) = ii;
           disp(sprintf('Ignoring test: %s',tests_todo{ii}))
       end
       [~,idx]=find(res>0);
       ignoreUsed(end+1:end+numel(idx)) = idx;
    end
    
    if ~isempty(ignoreList)
       tests_todo(ignoreList) = []; 
    end
    
    if numel(ignoreUsed)~=numel(ignoreCell)
        ignoreCell(ignoreUsed) = [];
        strToPlot = cellfun(@(iEl) [iEl,', '],ignoreCell,'UniformOutput',0);
        strToPlot = cell2mat(strToPlot);  
        
        error('%s: The following ignored tests were not found: %s',...
              upper(mfilename),strToPlot(1:end-2));

    end
end


precarray={'double','single'};
if nargin >0 && ~strcmpi(prec,'all')
  if any(cellfun(@(pEl)strcmpi(pEl,prec),precarray))  
    precarray={prec}; 
  else
    error('%s: Unknown data precision.',upper(mfilename));  
  end
end

total_tests_failed=0;
list_of_failed_tests={};

for precidx=1:numel(precarray)
    prec=precarray{precidx};

    for ii=1:length(tests_todo)
        test_failed=feval(['test_',tests_todo{ii}],prec);
        total_tests_failed=total_tests_failed+test_failed;
        if test_failed>0
            list_of_failed_tests{end+1}=['test_',tests_todo{ii},' ',prec];
        end;
    end;
end;

disp(' ');
if total_tests_failed==0
  disp('ALL TESTS PASSED');
else
  s=sprintf('%i TESTS FAILED',total_tests_failed);
  disp(s);
  disp('The following test scripts contained failed tests');
  for ii=1:length(list_of_failed_tests)
    disp(['   ',list_of_failed_tests{ii}]);
  end;
end;


%
%   Url: http://ltfat.github.io/doc/libltfat/modules/libphaseret/testing/mUnit/test_all_libphaseret.html

% Copyright (C) 2005-2018 Peter L. Soendergaard <peter@sonderport.dk> and others.
% This file is part of LTFAT version 2.4.0
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

