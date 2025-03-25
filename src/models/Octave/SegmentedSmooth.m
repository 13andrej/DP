function SmoothedSignal=SegmentedSmooth(y,smoothwidths,type,ends)
%   SegmentedSmooth(y,w,type,ends) divides y into a number of equal-length
%   segments defined by the length of the vector 'smoothwidths', then
%   smooths each segment with smooth of type 'type' and width defined by
%   the elements of vector 'smoothwidths'.
%    Version 1, November 2016.
%   The argument "type" determines the smooth type:
%     If type=1, rectangular (sliding-average or boxcar)
%     If type=2, triangular (2 passes of sliding-average)
%     If type=3, pseudo-Gaussian (3 passes of sliding-average)
%     If type=4, pseudo-Gaussian (4 passes of same sliding-average)
%     If type=5, multiple-width (4 passes of different sliding-averages)
%   The argument "ends" controls how the "ends" of the signal
%   (the first w/2 points and the last w/2 points) are handled.
%     If ends=0, the ends are zero. (In this mode the elapsed
%       time is the fastest and is independent of the smooth width).
%     If ends=1, the ends are smoothed with progressively
%       smaller smooths the closer to the end. (In this mode the
%       elapsed time increases with increasing smooth widths).
%   SegmentedSmooth(Y,w,type) smooths with ends=0.
%   SegmentedSmooth(Y,w) smooths with type=1 and ends=0.
%
%   Examples: 3-segment smooth of random white noise, smooth widths of
%   2,20, and 200.
%     x=1:10000;y=randn(size(x));
%     plot(x,SegmentedSmooth(y,[2 20 200],3,0))
%
%     20-segment smooth, odd smooth widths from 1 to 41:
%     plot(x,SegmentedSmooth(y,[1:2:41],3,0))

%   Copyright (c) 2012, Thomas C. O'Haver
if nargin==2, ends=0; type=1; end
if nargin==3, ends=0; end
ly=length(y);
NumSegments=length(smoothwidths);
SegLength=round(ly./NumSegments);
SmoothSegment=zeros(ly,NumSegments);
SmoothedSignal=zeros(1,ly);
for Segment=1:NumSegments,
    SmoothSegment(:,Segment)=fastsmooth(y,smoothwidths(Segment),type,ends);
    startindex=(1+(Segment-1)*SegLength);
    endindix=startindex+SegLength-1;
    if endindix>ly,endindix=ly;end
    indexrange=startindex:endindix;
    SmoothedSignal(indexrange)=SmoothSegment(indexrange,Segment);
end
