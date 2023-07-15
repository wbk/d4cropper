use std::env;
use std::cmp::{min, max};
use chrono::Local;
use directories;
use opencv::{
    core::{self, min_max_loc, Mat, Point, Size, Rect, no_array, Vector},
    imgproc::{match_template, TemplateMatchModes, resize},
    imgcodecs::{imread, imwrite},
    types::VectorOfMat,
    prelude::*,
};
use opener;

fn match_template_position(source: &Mat, template_path: &str) -> opencv::Result<Rect> {
    let mat_template = imread(template_path, 1)?;

    let mut result = Mat::default();
    match_template(&source, &mat_template, &mut result, TemplateMatchModes::TM_CCORR_NORMED as i32, &no_array())?;

    let mut min_val = 0f64;
    let mut max_val = 0f64;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    min_max_loc(&result, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &no_array())?;

    let template_size = Size::new(mat_template.cols(), mat_template.rows());
    let roi = Rect::new(max_loc.x, max_loc.y, template_size.width, template_size.height);

    Ok(roi)
}

fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Please provide at least one argument: <source image path>");
        std::process::exit(1);
    }

    let num_columns = match args.len() {
        5 => 2,
        6 => 3,
        7 => 3,
        8 => 4,
        _ => 6
    };

    let mut cropped_images = vec![];
    let mut rows = vec![];
    let mut max_width = 0;
    let mut max_height = 0;
    let mut tiles = VectorOfMat::new();

    for (i, arg) in args.iter().skip(1).enumerate() {
        let source_path = arg;
        let mat_source = imread(source_path, 1)?;

        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;

        let templates = ["tl.png", "tr.png", "bl.png", "br.png"];
        for template in &templates {
            let roi = match_template_position(&mat_source, &format!("corner_templates/{}", template))?;
            min_x = min(min_x, roi.x);
            min_y = min(min_y, roi.y);
            max_x = max(max_x, roi.x + roi.width);
            max_y = max(max_y, roi.y + roi.height);
        }

        let roi = Rect::new(min_x, min_y, max_x - min_x, max_y - min_y);
        let mat_cropped = Mat::roi(&mat_source, roi)?;

        max_height = max(max_height, mat_cropped.rows());

        cropped_images.push(mat_cropped);

        if (i + 1) % num_columns == 0 || i == args.len() - 2 {
            for mat_cropped in cropped_images.drain(..) {
                let mut mat_resized = mat_cropped.clone();
                let scale = max_height as f64 / mat_cropped.rows() as f64;
                let width_scaled = (mat_cropped.cols() as f64 * scale) as i32;
                resize(&mat_cropped, &mut mat_resized, Size::new(width_scaled, max_height), 0.0, 0.0, 1)?;
                tiles.push(mat_resized);
            }

            let mut row = Mat::default();
            core::hconcat(&tiles, &mut row)?;

            max_width = max(max_width, row.cols());
            rows.push(row);

            tiles.clear();
            max_height = 0;
        }
    }

    for row in &mut rows {
        if row.cols() < max_width {
            let padding = Mat::new_rows_cols_with_default(row.rows(), max_width - row.cols(), row.typ(), core::Scalar::all(0.0))?;
            let mut padded_row = Mat::default();
            let mut concatenate = VectorOfMat::new();
            concatenate.push(row.clone());
            concatenate.push(padding);
            core::hconcat(&concatenate, &mut padded_row)?;
            *row = padded_row;
        }
    }

    let mut result = Mat::default();
    let mut rows_vec = VectorOfMat::new();
    for row in rows {
        rows_vec.push(row);
    }
    core::vconcat(&rows_vec, &mut result)?;

    let date = Local::now();
    let path =directories::UserDirs::new().unwrap().picture_dir().unwrap().join(format!("tile-{}.jpg", date.format("%Y-%m-%d-%H-%M-%S")));
    imwrite(path.to_str().unwrap(), &result, &Vector::new())?;
    opener::open(path).unwrap();

    Ok(())
}