require 'gnuplot'


function plot_mointor_err(fn_err, fn_img)
  local plot_script_template = [[
    set term svg
    set output "%s"
    plot "%s" using %d:%d with linespoints linecolor 1 linewidth 2 pointtype 7 pointsize 2 
    set title "%s"
    set xlabel "iter"
    set ylabel "loss"
    set grid
    set output 
  ]]
  
  local cmd = string.format(plot_script_template, fn_img, fn_err, 1, 3, 'training process')
  gunplot.raw(cmd)
end

function plot_monitor_errs(fn_err, fn_exp, fn_img, fn_imgs)
  
  
  for i = 1, #fn_imgs do
    cmd = string.format(plot_script_template, fn_imgs[i], fn_exp, 1, i+2, string.format('example %d', i) )
  end
end
