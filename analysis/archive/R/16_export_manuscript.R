# 16_export_manuscript.R
# Copies exactly the figures/tables that manuscript/main.tex uses into
# manuscript/figures and manuscript/tables, checks every \includegraphics and
# \input target exists, and zips manuscript/ as manuscript_overleaf.zip --
# upload that via Overleaf "New Project > Upload Project" (a fresh project, so
# nobody's live edits in the shared project get touched).

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))

MS_DIR <- file.path(PIPELINE_ROOT, "manuscript")
tex <- readLines(file.path(MS_DIR, "main.tex"))
tex_nocomment <- sub("(^|[^\\\\])%.*$", "\\1", tex)

figs <- unlist(regmatches(tex_nocomment, gregexpr("\\\\includegraphics(\\[[^]]*\\])?\\{[^}]+\\}", tex_nocomment)))
figs <- sub(".*\\{([^}]+)\\}$", "\\1", figs)
tabs <- unlist(regmatches(tex_nocomment, gregexpr("\\\\input\\{[^}]+\\}", tex_nocomment)))
tabs <- sub(".*\\{([^}]+)\\}$", "\\1", tabs)
tabs <- ifelse(grepl("\\.tex$", tabs), tabs, paste0(tabs, ".tex"))
log_msg("16", sprintf("main.tex references %d figures and %d tables", length(figs), length(tabs)))

dir.create(file.path(MS_DIR, "figures"), showWarnings = FALSE)
dir.create(file.path(MS_DIR, "tables"), showWarnings = FALSE)
for (f in figs) {
  src <- file.path(OUT_FIGURES, basename(f))
  assert_that(file.exists(src), sprintf("main.tex needs %s but the pipeline did not produce it", src))
  file.copy(src, file.path(MS_DIR, f), overwrite = TRUE)
}
for (t in tabs) {
  src <- file.path(OUT_TABLES, basename(t))
  assert_that(file.exists(src), sprintf("main.tex needs %s but the pipeline did not produce it", src))
  file.copy(src, file.path(MS_DIR, t), overwrite = TRUE)
}

zip_path <- file.path(PIPELINE_ROOT, "manuscript_overleaf.zip")
if (file.exists(zip_path)) file.remove(zip_path)
# utils::zip() needs a system `zip` binary (present on macOS and the cloud
# container, absent in the Cowork sandbox). Without it, skip the zip loudly --
# manuscript/ itself is complete and can be zipped by hand.
if (nzchar(Sys.which("zip"))) {
  old <- setwd(MS_DIR)
  extra <- intersect(c("cover_letter.tex", "main_pnas.tex"), list.files("."))  # Science cover letter; PNAS fallback version
  utils::zip(zip_path, c("main.tex", extra, figs, tabs), flags = "-q")
  setwd(old)
  log_msg("16", "Wrote ", zip_path)
} else {
  log_msg("16", "WARNING: no `zip` binary on this machine -- manuscript/ is up to date but manuscript_overleaf.zip was NOT rebuilt")
}
