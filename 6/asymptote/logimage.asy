if(!settings.multipleView) settings.batchView=false;
settings.tex="pdflatex";
defaultfilename="logimage";
if(settings.render < 0) settings.render=4;
settings.outformat="";
settings.inlineimage=true;
settings.embed=true;
settings.toolbar=false;
viewportmargin=(2,2);

texpreamble("\include{commands}");
texpreamble("\usepackage{mathptmx}");
texpreamble("\usepackage[T1]{fontenc}");
texpreamble("\usepackage{amssymb}");
texpreamble("\usepackage{amsmath}");
texpreamble("\usepackage{mathtools}");
texpreamble("\usepackage{upgreek}");







import graph;
import palette;

size(10cm,10cm,IgnoreAspect);

real f(real x, real y) {
  return 0.9*pow10(2*sin(x/5+2*y^0.25)) + 0.1*(1+cos(10*log(y)));
}

scale(Linear,Log,Log);

pen[] Palette=BWRainbow();

bounds range=image(f,Automatic,(0,1),(100,100),nx=200,Palette);

xaxis("$x$",BottomTop,LeftTicks,above=true);
yaxis("$y$",LeftRight,RightTicks,above=true);

palette("$f(x,y)$",range,(0,200),(100,250),Top,Palette,
        PaletteTicks(ptick=linewidth(0.5*linewidth())));

