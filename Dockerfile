FROM jekyll/jekyll:4
COPY . /srv/jekyll/
RUN  bundle && jekyll build
ENTRYPOINT [ "jekyll", "serve", "--incremental", "--watch"]
EXPOSE 4000