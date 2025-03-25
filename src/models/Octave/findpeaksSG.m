function P=findpeaksSG(x,y,SlopeThreshold,AmpThreshold,smoothwidth,peakgroup,smoothtype)
% Segmented peak finder, same syntax as findpeaksG except 3rd to 6th inut
% arguments can be vectors with ine entry for each segment.
% function P=findpeaksSG(x,y,SlopeThresholds,AmpThresholds,smoothwidths,peakgroups,smoothtype)
% Locates and measures the positive peaks in a noisy x-y time series data.
% Detects peaks by looking for downward zero-crossings in the first
% derivative whose upward slopes exceed SlopeThreshold. Returns list (P)
% containing peak number and position, height, width, and area of each
% peak. Arguments "slopeThreshold", "ampThreshold" and "smoothwidth"
% control peak sensitivity of each segment. Higher values will neglect
% smaller features. "Smoothwidth" is a vector of the widths of the smooths
% applied before peak detection; larger values ignore narrow peaks. If
% smoothwidth=0, no smoothing is performed. "Peakgroup" is a vector of the
% number points around the top part of the peak that are taken for
% measurement. If Peakgroup=0 the local maximum is taken as the peak height
% and position. The argument "smoothtype" determines the smooth algorithm:
%   If smoothtype=1, rectangular (sliding-average or boxcar) If
%   smoothtype=2, triangular (2 passes of sliding-average) If smoothtype=3,
%   pseudo-Gaussian (3 passes of sliding-average)
% See http://terpconnect.umd.edu/~toh/spectrum/Smoothing.html and
% http://terpconnect.umd.edu/~toh/spectrum/PeakFindingandMeasurement.htm
% (c) T.C. O'Haver, 2016.  Version 1, November, 2016
%
% Example: Find, measure, and plot noisy peaks with very different widths
%   x=1:.2:100;
%   y=gaussian(x,20,1.5)+gaussian(x,80,30)+.02.*randn(size(x));
%   plot(x,y,'c.')
%   P=findpeaksSG(x,y,[0.001 .0001],[.2 .2],[5 10],[10 50],3)
%   text(P(:,2),P(:,3),num2str(P(:,1)))
%   disp(' ')
%   disp('           peak #    Position      Height     Width      Area')
%   disp(P)
%
% Related functions:
% findpeaksG.m, findvalleys.m, findpeaksL.m, findpeaksb.m, findpeaksb3.m,
% findpeaksplot.m, peakstats.m, findpeakSNR.m, findpeaksGSS.m,
% findpeaksLSS.m, findpeaksfit.m, findsteps.m, findsquarepulse.m, idpeaks.m
% Copyright (c) 2016 Thomas C. O'Haver
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.
%
if nargin~=7;smoothtype=1;end  % smoothtype=1 if not specified in argument
if smoothtype>3;smoothtype=3;end
if smoothtype<1;smoothtype=1;end
if smoothwidth<1;smoothwidth=1;end
if isscalar(AmpThreshold),AmpThreshold=AmpThreshold.*ones(size(SlopeThreshold));end
if isscalar(smoothwidth),smoothwidth=smoothwidth.*ones(size(SlopeThreshold));end
if isscalar(peakgroup),peakgroup=peakgroup.*ones(size(SlopeThreshold));end
smoothwidth=round(smoothwidth);
peakgroup=round(peakgroup);
if smoothwidth>1,
    d=SegmentedSmooth(deriv(y),smoothwidth,smoothtype);
else
    d=deriv(y);
end
P=[0 0 0 0 0];
vectorlength=length(y);
NumSegs=length(SlopeThreshold);
peak=1;
for j=2*round(smoothwidth/2)-1:length(y)-smoothwidth-1,
    Seg=fix(1+NumSegs./(vectorlength./j));
    n=round(peakgroup(Seg)/2+1);
    % [j Seg]
    if sign(d(j)) > sign (d(j+1)), % Detects zero-crossing
        if d(j)-d(j+1) > SlopeThreshold(Seg), % if slope of derivative is larger than SlopeThreshold
            if y(j) > AmpThreshold(Seg),  % if height of peak is larger than AmpThreshold
                xx=zeros(size(peakgroup(Seg)));yy=zeros(size(peakgroup(Seg)));
                for k=1:peakgroup(Seg), % Create sub-group of points near peak
                    groupindex=j+k-n+2;
                    if groupindex<1, groupindex=1;end
                    if groupindex>vectorlength, groupindex=vectorlength;end
                    % groupindex=groupindex
                    xx(k)=x(groupindex);
                    yy(k)=y(groupindex);
                end
                if peakgroup(Seg)>2,
                    [Height,Position,Width]=gaussfit(xx,yy);
                    PeakX=real(Position);   % Compute peak position and height of fitted parabola
                    PeakY=real(Height);
                    MeasuredWidth=real(Width);
                    % if the peak is too narrow for least-squares technique to work
                    % well, just use the max value of y in the sub-group of points near peak.
                else
                    PeakY=max(yy);
                    pindex=val2ind(yy,PeakY);
                    PeakX=xx(pindex(1));
                    MeasuredWidth=0;
                end
                % Construct matrix P. One row for each peak detected,
                % containing the peak number, peak position (x-value) and
                % peak height (y-value). If peak measurement fails and
                % results in NaN, or if the measured peak height is less
                % than AmpThreshold, skip this peak
                if isnan(PeakX) || isnan(PeakY) || PeakY<AmpThreshold(Seg),
                    % Skip this peak
                else % Otherwise count this as a valid peak
                    P(peak,:) = [round(peak) PeakX PeakY MeasuredWidth  1.0646.*PeakY*MeasuredWidth];
                    peak=peak+1; % Move on to next peak
                end
            end
        end
    end
end
